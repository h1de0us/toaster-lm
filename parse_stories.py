import os
import json

def collect_data_from_folder(folder_path):
    all_stories = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    for entry in data:
                        if "story" in entry:
                            story_text = entry["story"]
                            all_stories.append(story_text)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {filename}: {e}")
            os.remove(file_path)
    print(len(all_stories))
    return all_stories

def save_stories_to_file(stories, output_file="stories.txt"):
    with open(output_file, "w", encoding="utf-8") as output:
        for story in stories:
            story = story.replace("\n", " ") # if we want to fit a story on one line
            output.write(story + "\n") # remove adding "\n" if you split a story into several lines

folder_path = "data"

all_stories = collect_data_from_folder(folder_path)

save_stories_to_file(all_stories, output_file="stories.txt")
