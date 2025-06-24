<?php

// --- 1. CONFIGURATION AND SETUP ---

// Define the path to the data file. Using a constant makes it easy to change if needed.
define('DUTIES_FILE_PATH', 'EDEntryV2.txt');
// Set the default timezone to ensure date calculations are consistent.
date_default_timezone_set('NZ');


// --- 2. CORE FUNCTIONS ---

/**
 * Reads duties from a flat file and returns them as an array of structured objects.
 * Each object represents a single task or duty.
 *
 * @param string $filePath The path to the data file.
 * @return array An array of duty objects. Returns an empty array on failure.
 */
function getDutiesFromFile(string $filePath): array
{
    // Check if the file exists and is readable to prevent errors.
    if (!file_exists($filePath) || !is_readable($filePath)) {
        // In a real application, you might log this error.
        return [];
    }

    $duties = [];
    $fileContent = file($filePath, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);

    foreach ($fileContent as $line) {
        // Split each line by the '|' delimiter.
        $parts = explode('|', $line, 4); // Limit to 4 parts to handle extra '|' in comments.
        
        // Ensure the line has the expected number of parts before processing.
        if (count($parts) === 4) {
            $duties[] = (object)[
                'dueDate'   => trim($parts[0]),
                'comment'   => trim($parts[1]),
                'assignee'  => trim($parts[2]),
                'status'    => trim($parts[3]),
            ];
        }
    }
    
    return $duties;
}

/**
 * Processes an array of duties: sorts them by date and adds styling information.
 *
 * @param array $duties The raw array of duty objects.
 * @return array The processed and sorted array of duties.
 */
function processAndSortDuties(array $duties): array
{
    // Sort the duties by due date, with the most recent first.
    usort($duties, function ($a, $b) {
        return strtotime($b->dueDate) <=> strtotime($a->dueDate); // Spaceship operator for comparison
    });

    $todayTimestamp = time();
    $sevenDaysInSeconds = 7 * 86400;

    // Loop through the sorted duties once to add display logic.
    foreach ($duties as $duty) {
        $dutyTimestamp = strtotime($duty->dueDate);
        $timeDifference = $dutyTimestamp - $todayTimestamp;

        // --- Determine CSS classes for styling ---
        $duty->rowClass = 'task-default';
        $duty->dateClass = 'date-normal';

        // Apply 'completed' styling if the task is done or cancelled.
        if (in_array($duty->status, ['Completed', 'Done', 'Cancelled'])) {
            $duty->rowClass = 'task-completed';
        } else {
            // If not completed, check if it's due soon.
            if ($timeDifference >= 0 && $timeDifference < $sevenDaysInSeconds) {
                $duty->dateClass = 'date-due-soon';
            }
        }
        
        // --- Sanitize output and format comment ---
        // IMPORTANT: Escape all user-provided data before rendering to prevent XSS attacks.
        $duty->dueDateEscaped = htmlspecialchars($duty->dueDate, ENT_QUOTES, 'UTF-8');
        $duty->assigneeEscaped = htmlspecialchars($duty->assignee, ENT_QUOTES, 'UTF-8');
        $duty->statusEscaped = htmlspecialchars($duty->status, ENT_QUOTES, 'UTF-8');

        // Make URLs in the comment clickable, after escaping the rest of the comment text.
        $escapedComment = htmlspecialchars($duty->comment, ENT_QUOTES, 'UTF-8');
        $duty->commentFormatted = preg_replace(
            '/(https?:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?)/',
            '<a href="$1" target="_blank" rel="nofollow">$1</a>',
            $escapedComment
        );
    }
    
    return $duties;
}


// --- 3. DATA PREPARATION FOR RENDERING ---

// Execute the functions to get the final data ready for the HTML view.
$allDuties = getDutiesFromFile(DUTIES_FILE_PATH);
$processedDuties = processAndSortDuties($allDuties);

?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extra Duties Notice Board</title>
    <!-- Link to the external CSS file for styling. -->
    <link href="css/nb_styles.css" rel="stylesheet">
</head>
<body>

    <div class="container">
        <h1>Extra Duties Notice Board</h1>
        <hr />

        <table class="notice-board">
            <thead>
                <tr>
                    <th class="col-date">Due Date</th>
                    <th class="col-comment">Comment</th>
                    <th class="col-assignee">Assigned To</th>
                    <th class="col-status">Status</th>
                </tr>
            </thead>
            <tbody>
                <?php if (empty($processedDuties)): ?>
                    <tr>
                        <td colspan="4">No duties found or the data file is unavailable.</td>
                    </tr>
                <?php else: ?>
                    <!-- Loop through the processed data and create a table row for each duty. -->
                    <?php foreach ($processedDuties as $duty): ?>
                        <tr class="<?= $duty->rowClass; ?>">
                            <td class="<?= $duty->dateClass; ?>"><?= $duty->dueDateEscaped; ?></td>
                            <td><?= $duty->commentFormatted; // Already escaped and formatted ?></td>
                            <td><?= $duty->assigneeEscaped; ?></td>
                            <td><?= $duty->statusEscaped; ?></td>
                        </tr>
                    <?php endforeach; ?>
                <?php endif; ?>
            </tbody>
        </table>

        <div class="footer-links">
            <p>
                To assign a new duty, please visit the 
                <a href="http://iforecast.rf.gd/MET/CNB/ExtraDutiesEntryFormV2.php" target="_blank">Entry Form</a>.
            </p>
            <p>
                To edit an existing duty, please visit the 
                <a href="http://iforecast.rf.gd/MET/CNB/ExtraDutiesEditFormV2.php" target="_blank">Edit Form</a>.
            </p>
        </div>
    </div>

</body>
</html>
