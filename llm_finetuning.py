# Description:
# This script fine-tunes a T5-base model on a custom weather forecast dataset.
# It includes a full pipeline: data loading, training, validation using ROUGE-L score,
# and a robust checkpointing system to save progress after each epoch.
# The script is designed to be resumable, making it suitable for environments with time limits like Google Colab.
#

# --- 1. SETUP AND IMPORTS ---

# Install required libraries for this notebook environment.
# In a production environment, these would be managed in a requirements.txt file.
!pip install transformers
!pip install sentencepiece
!pip install rouge-score

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from rouge_score import rouge_scorer
from tqdm import tqdm  # For displaying progress bars
from google.colab import drive # Specific to Google Colab for file access

# --- 2. DATA PREPARATION ---

# Mount Google Drive to access files stored there.
drive.mount('/content/drive')

# Initialize the T5 tokenizer, which is responsible for converting text into a format the model understands.
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Define a custom PyTorch Dataset for loading and tokenizing the weather forecast data on-the-fly.
class WeatherForecastDataset(Dataset):
    """
    Custom Dataset to handle tokenization of input data and target forecasts.
    It takes a dataframe with 'input_text' and 'target_text' columns.
    """
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.input_texts = data['input_text'].tolist()
        self.target_texts = data['target_text'].tolist()
        self.max_length = max_length

    def __len__(self):
        # Returns the total number of samples in the dataset.
        return len(self.input_texts)

    def __getitem__(self, index):
        # Fetches a single sample and tokenizes it for the model.
        input_text = self.input_texts[index]
        target_text = self.target_texts[index]

        # The tokenizer converts the text to a sequence of numbers (input_ids).
        # Padding ensures all sequences in a batch have the same length.
        # Truncation cuts off sequences that are longer than max_length.
        input_ids = self.tokenizer.encode(input_text,
                                          max_length=self.max_length,
                                          padding='max_length',
                                          truncation=True,
                                          return_tensors="pt").squeeze() # .squeeze() removes unnecessary dimensions.
        target_ids = self.tokenizer.encode(target_text,
                                           max_length=self.max_length,
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors="pt").squeeze()

        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }

# Load a pre-processed and tokenized dataset to save time during execution.
dataset = torch.load('/content/drive/MyDrive/WxLangModel/TokenizedData_V15_T5base.pth')

# Split the full dataset into a training set (90%) and a validation set (10%).
# The model learns from the training set and its performance is measured on the unseen validation set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for handling batching and shuffling of the data.
# Shuffling the training data is important to prevent the model from learning the order of the data.
train_loader = DataLoader(train_dataset, batch_size=14, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=14, shuffle=False)


# --- 3. MODEL CONFIGURATION AND CHECKPOINTING ---

# Define base paths for saving the model and checkpoints.
# Using a clear naming convention helps in managing different experiment versions.
MODEL_BASE_PATH = '/content/drive/MyDrive/WxLangModel/AdaSpecModel/TrainedModel_V15_T5base'
CHECKPOINT_BASE_PATH = '/content/drive/MyDrive/WxLangModel/AdaSpecModel/Checkpoint_V15_T5base'

# Check if a GPU is available and set the device accordingly. Training is significantly faster on a GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained T5-base model from Hugging Face and move it to the selected device.
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.to(device)

# Define the optimizer. Adafactor is memory-efficient and often works well for Transformer models.
# The parameters here are set to mirror the behavior of Adam, but without storing momentum.
optimizer = Adafactor(model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)

# Define the learning rate scheduler. StepLR reduces the learning rate by a factor (gamma) every step_size epochs.
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# Set the total number of training epochs.
num_epochs = 30

# --- Checkpoint Loading Logic ---
# This block makes the script resumable. It finds the latest saved checkpoint and loads its state.
start_epoch = 0
latest_checkpoint_path = None
# Iterate backwards from the last possible epoch to find the most recent checkpoint that exists.
for epoch in reversed(range(num_epochs)):
    checkpoint_path = f"{CHECKPOINT_BASE_PATH}_Epoch_{epoch + 1}.pth"
    if os.path.exists(checkpoint_path):
        latest_checkpoint_path = checkpoint_path
        start_epoch = epoch + 1 # We will start from the *next* epoch.
        break # Exit the loop once the latest checkpoint is found.

if latest_checkpoint_path:
    print(f"Resuming training. Loading checkpoint from {latest_checkpoint_path}...")
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Successfully loaded checkpoint. Starting from Epoch {start_epoch + 1}.")
else:
    print("No checkpoint found. Starting training from scratch.")

# --- 4. TRAINING AND VALIDATION LOOP ---

# This flag is a pragmatic solution for Google Colab, allowing the user to
# run one epoch at a time and manually restart, preventing runtime disconnections on long jobs.
stop_training = False

# The main training loop iterates for the specified number of epochs.
for epoch in range(start_epoch, num_epochs):
    print(f"--- Starting Epoch {epoch + 1}/{num_epochs} ---")

    # --- Training Phase ---
    model.train() # Set the model to training mode (enables dropout, etc.).
    total_train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        # Move the data batch to the GPU/CPU.
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        # Reset gradients from the previous step.
        optimizer.zero_grad()

        # Forward pass: compute the model's output and loss.
        # The 'labels' argument tells the model to compute the loss internally.
        output = model(input_ids=input_ids, labels=target_ids)
        loss = output.loss
        total_train_loss += loss.item()

        # Backward pass: compute gradients.
        loss.backward()

        # Update the model's weights.
        optimizer.step()

    # Step the learning rate scheduler after each epoch.
    scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode (disables dropout, etc.).
    rouge_l_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # ROUGE-L is good for summary-style tasks.
    
    # torch.no_grad() disables gradient calculation, saving memory and speeding up validation.
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Generate predictions from the model.
            output_ids = model.generate(input_ids, max_length=80, num_beams=4, early_stopping=True)

            # Decode the generated IDs back into human-readable text.
            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

            # Calculate ROUGE-L F-measure for each prediction against its reference.
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                rouge_l_scores.append(score['rougeL'].fmeasure)

    # Calculate the average ROUGE-L score across the entire validation set.
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    print(f"Epoch {epoch + 1} - Average ROUGE-L F-measure: {avg_rouge_l:.4f}")

    # --- 5. SAVE MODEL AND CHECKPOINT ---
    
    # Define unique paths for this epoch's checkpoint and model artifacts.
    epoch_checkpoint_path = f"{CHECKPOINT_BASE_PATH}_Epoch_{epoch + 1}.pth"
    epoch_model_save_path = f"{MODEL_BASE_PATH}_Epoch_{epoch + 1}"

    # Save a checkpoint dictionary containing model, optimizer, and scheduler states.
    # This allows for exact resumption of the training state.
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'validation_rouge_l': avg_rouge_l # Also save the validation score for tracking.
    }, epoch_checkpoint_path)

    # Save the model itself in the Hugging Face format.
    model.save_pretrained(epoch_model_save_path)
    print(f"Checkpoint and model saved for Epoch {epoch + 1}")

    # --- Control Flow for Colab ---
    # This block ensures the script stops after one successful epoch.
    if epoch + 1 < num_epochs:
        print(f"Epoch {epoch + 1} completed. To continue training, please restart the script.")
        stop_training = True
    else:
        print("Final epoch completed. Training finished.")

    if stop_training:
        break # Exit the main training loop.

print("\n--- Training script finished. ---")
if stop_training:
    print("The script was halted intentionally after one epoch. Please manually restart for the next epoch.")
