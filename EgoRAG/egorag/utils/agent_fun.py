import re
from egorag.utils.util import call_gpt4
def get_ids(question, results):
    system_prompt = """
    You are an AI assistant tasked with helping to process and analyze search results for specific questions. 
    When a user provides a question along with a set of search results, your job is to:

    1. Review all search results.
    2. Find the most recent entry that is directly related to the user's question.
    3. Consider both the relevance of the document and the time of the event (based on the provided timestamp).
    4. Return the `id` of the most relevant and latest document.
    5. Output the id with format 'ID: <id>'

    Please ensure that you consider both the content of the document and the timestamp to identify the most recent and relevant document.

    Your responses should only include the format `ID: <id>` of the selected document.

    Here is an example to guide your process:

    **Question:**  
    "When was the last time the guitar was played?"

    **Search Results:**  
    [
        {'id': 'DAY2-16170000-16173000_5', 'document': ['Katrina mentioned that the projector on the second floor was nice too and asked if we could play some music.', 'I agreed, "Play some music.', '" Then I placed the box Lucia handed me onto the desk, opened it, looked at Lucia, and then closed the box.'], 'date': 2, 'end_time': 16173000, 'distance': 0.47747164964675903}, 
        {'id': 'DAY2-18163000-18170000_2', 'document': ['As we talked, he scratched his nose, which made me chuckle.', 'We were having a lively conversation, and he even started playing the guitar.', "I mentioned, 'Ideally, we should have a script supervisor, it would make the work much easier."], 'date': 2, 'end_time': 18170000, 'distance': 0.5553568601608276}, 
        {'id': 'DAY2-18200000-18203000_3', 'document': ["I said, 'The An and Ankang fish are collected.", "' Then I tilted my head and continued chatting with Choiszt, who was strumming his guitar while responding to me.", "I said, '20 minutes, 20 minutes down,' and added, 'That's fine, one more hour, right?' At that moment, I noticed Nicous beside me, touching his hair."], 'date': 2, 'end_time': 18203000, 'distance': 0.616065502166748}, 
        {'id': 'DAY2-18270000-18273000_4', 'document': ['Sitting on the bed with my legs bent, I glanced at my colleagues.', 'Some were busy at their computers, a boy was playing the guitar, and others were focused on their tasks.', 'I observed the scattered electronics and messy bedding, contemplating whether to join their discussion.'], 'date': 2, 'end_time': 18273000, 'distance': 0.6114476323127747}, 
        {'id': 'DAY3-16163000-16170000_5', 'document': ["I put down the box and bottle in my hands and hesitantly said, 'Seems like not.", "Yeah, then I'll start playing.", "' The constantly changing scores on the screen confused me, so I asked, 'What does that mean?' At that moment, Tasha asked, 'Can't we play together?' I replied, 'Sure, you go first then."], 'date': 3, 'end_time': 16170000, 'distance': 0.6253791451454163}, 
        {'id': 'DAY3-21160000-21163000_3', 'document': ["We were chatting happily, and Lucia laughed, saying, 'They might just be simply moving the dishes from one place to another.", "' I joked, 'Playing music, hahaha, continue.", "' Then I said, 'Alright, alright, even though we can't hit it, keep playing the music."], 'date': 3, 'end_time': 21163000, 'distance': 0.604546308517456}, 
        {'id': 'DAY3-22053000-22060000_3', 'document': ["Tasha asked, 'Hey, didn't you play that the other day?' to which I replied, 'Yeah, it's good.", 'You gotta play well, gotta play.', "' Just then, I saw Tasha and Katrina coming over, dressed in casual clothes with happy smiles on their faces."], 'date': 3, 'end_time': 22060000, 'distance': 0.5654604434967041}, 
        {'id': 'DAY3-22453000-22460000_3', 'document': ['Lucia also came over, and Shure was looking into my eyes.', 'I moved closer to see him as he played guitar and sang.', 'I heard him sing, "I really hope you can muster up some courage.'], 'date': 3, 'end_time': 22460000, 'distance': 0.49885010719299316}, 
        {'id': 'DAY3-22460000-22463000_4', 'document': ["' I noticed the table was filled with food and drinks, and everyone seemed very relaxed.", "Shure was playing the guitar, and I nodded my head to the rhythm; his music was so beautiful and moving that I couldn't help but sway gently.", 'Lucia and Tasha were also eating at the table, while Alice continued to grill food, with the room dimly lit and the grill emitting fragrant smoke.'], 'date': 3, 'end_time': 22463000, 'distance': 0.6172605156898499}, 
        {'id': 'DAY3-22463000-22470000_0', 'document': ['I saw Shure playing the guitar and nodded to the rhythm of the music; the surrounding environment was very quiet.', 'Shure was playing happily, and the atmosphere was relaxed.'], 'date': 3, 'end_time': 22470000, 'distance': 0.6303433179855347}
    ]

    **Explanation of Fields in Search Results:**

    - **id**: This represents the time when the event described in the document occurred. The format is `DAYx_HHMMSSFF_HHMMSSFF`, where the first timestamp is the start time and the second is the end time of the event.
    - **document**: This is the textual description of the event that happened. It tells you what occurred at the given time.
    - **distance**: This indicates how similar the document is to the search query. The smaller the distance, the higher the relevance or similarity between the query and the document.
    - **date**: This represents the day on which the event described in the document occurred. It is a single integer (e.g., 1, 2, 3, etc.), indicating the specific day (e.g., Day 1, Day 2, etc.) relative to a defined starting point or timeline. Unlike end_time, which includes a timestamp, date only specifies the day.
    - **end_time**: This is the timestamp representing the moment when the event described in the document ended. It is also in the format `HHMMSSFF`.

    All times are formatted as `HHMMSSFF`, where:
    - `HH` stands for hours (00-23),
    - `MM` stands for minutes (00-59),
    - `SS` stands for seconds (00-59),
    - `FF` represents the frame (00-20).

    Note:The search results are already arranged in chronological order, from the earliest to the latest.
    
    **Expected Output:**  
    "ID: DAY3-22463000-22470000_0"

    Explanation: The most recent action related to the guitar being played is recorded at `DAY3_22463000_22470000_0`, which occurred at `22463000` and directly answers the user's question about the last time the guitar was tuned or played.
    """

    prompt = f"Question:{question}\n Search Results:{results}"
    answer = call_gpt4(prompt=prompt, system_message=system_prompt, temperature=0.1)
    id_match = re.search(r"ID:\s*(DAY\d+-\d{8}-\d{8}_\d)", answer)
    if id_match:
        id = id_match.group(1)

    else:
        print("No ID found in the response.")
        id = None
    return id



def parse_video_map(video_map):
    parsed_data = []
    for key, video_path in video_map.items():
        key_parts = key.split("-")
        date = key_parts[0]
        start_time = int(key_parts[1])
        end_time = int(key_parts[2])

        match = re.search(r"_(\d{8})\.mp4$", video_path)
        if match:
            video_start_time = int(match.group(1))
        else:
            video_start_time = None

        match = re.search(r"\d+", date)
        int_date = int(match.group()) if match else None

        parsed_data.append(
            {
                "date": date,
                "int_date": int_date,
                "start_time": start_time,
                "end_time": end_time,
                "video_path": video_path,
                "video_id": key,
                "video_start_time": video_start_time,
            }
        )
    return parsed_data

def transform_timedict(time_dict):
    date_time_mapping = defaultdict(list)

    for entry in time_dict:
        date_time_mapping[entry["date"]].append(str(entry["time"]))

    for date in date_time_mapping:
        date_time_mapping[date] = sorted(date_time_mapping[date])
    return date_time_mapping