// Wait for the entire HTML document to be loaded before running the script.
document.addEventListener('DOMContentLoaded', () => {

    // --- 1. GET DOM ELEMENTS ---
    // It's efficient to get all necessary elements from the DOM once.
    // 'const' is used because these variables will not be reassigned.
    const form = document.getElementById('converter-form');
    const inputText = document.getElementById('input-text');
    const outputHtml = document.getElementById('output-html');

    // --- 2. THE CONVERSION LOGIC ---
    // This function contains the core logic for converting plain text to HTML.
    const convertTextToHtml = (text) => {
        // Handle empty input gracefully by returning an empty string.
        if (!text.trim()) {
            return '';
        }

        // --- Character Escaping (Security) ---
        // First, escape essential HTML characters to prevent accidental code injection.
        // This ensures that if the user types '<' or '>', it's treated as text, not an HTML tag.
        let escapedText = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');

        // --- Paragraph and Line Break Handling ---
        // Now, handle line breaks to create paragraphs and line breaks (<br>).
        // The logic is much cleaner using regular expressions.
        const withParagraphs = escapedText
            // Replace two or more newlines with a closing and new opening paragraph tag.
            .replace(/\n{2,}/g, '</p><p>')
            // Replace single newlines with a <br> tag for a line break.
            .replace(/\n/g, '<br>');

        // Finally, wrap the entire output in a single <p> tag.
        return `<p>${withParagraphs}</p>`;
    };

    // --- 3. EVENT LISTENER ---
    // The modern way to handle events. This listens for the form's 'submit' event.
    form.addEventListener('submit', (event) => {
        // Prevent the default form submission behavior, which would reload the page.
        event.preventDefault();

        // Get the current value from the input textarea.
        const rawText = inputText.value;

        // Call the conversion function to get the processed HTML.
        const htmlResult = convertTextToHtml(rawText);

        // Display the final HTML in the output textarea.
        outputHtml.value = htmlResult;
    });

});
