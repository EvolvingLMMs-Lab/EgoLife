
gen_event_prompt = """
"You will be provided with some descriptions. 
Merge events into one single event based on these descriptions. 
Do not include uncertain information, speculation, or divergent content. 
Do not describe the atmosphere or emotions. Dismiss those provided content that are abstract or ambiguous. 
If the descriptions mention names of people who interacted with 'me', make sure to retain this information. 
Directly provide the summarized main events without adding any additional remarks or explanations.
"""

get_day_prompt = """You will be provided with a JSON dataset containing summaries of events spanning multiple days.

        Given the question: **{question}**, your task is to determine the most relevant date from the dataset that can be used to answer the question.

        ### Considerations:
        - The reference time for the question is **{date}, {time}**.
        - You must carefully analyze the necessary time periods to derive an accurate response.
        - Apply logical reasoning and general knowledge to ensure an appropriate selection.
        - **Select only one date** that is most likely to contain the required information.
        - If the question contains keywords like "yesterday", "last night", "today", etc., you should make the selection based on the reference date.

        ### Output Format (Strictly Adhere):
        - Your response **must** follow this exact format: [X], where X represents the day number.
        - Example: [3] for DAY3.
        - **Do not** include any additional text, explanations, or formatting beyond the required output.
        """

get_hour_prompt = """I will provide you with event descriptions, including their start time and end time in the format HHMMSSFF.
        For the question "{question}", determine the most appropriate time range that can be used to answer the question.
        The question time is {date}, {time}.

        You must carefully analyze the time periods necessary to derive an answer, incorporating logical reasoning and commonsense understanding.
        For each question, follow these thinking steps:
        - Identify the key elements from the question.
        - Infer a relevant event based on the key elements.
        - Determine the most probable time range associated with the event.

        Examples:
        Q1: "What is the first ingredient in the shopping cart yesterday?"
        - You should first find the key elements: shopping cart, first ingredient.
        - Then you might conclude the relevant event: shopping activity.
        - Finally, you should determine the time range: likely during the shopping event.

        Q2: "When was the pizza placed on the table?"
        - You should first find the key elements: pizza, placed on the table.
        - Then you might conclude the relevant event: meal preparation or serving.
        - Finally, you should determine the time range: likely within the meal preparation or serving period.

        Q3: "Where were the flowers placed before?"
        - You should first find the key elements: flowers, placed before.
        - Then you might conclude the relevant event: movement of the flowers.
        - Finally, you should determine the time range: likely before the flowers were relocated.

        Now you understand how to think and determine the time range. But your response should provide a SINGLE time range, including both the start timestamp and end timestamp.
        Ensure that your output contains only the time range in the following format:
        [Start Time]
        [End Time]

        Example output:
        [11000000]
        [14000000]
        """
        
query_prompt = """
                You are an intelligent assistant designed to:
                1. Extract concise and relevant keywords from user-provided questions.
                2. Determine the search time range based on the options provided (if applicable).

                ## Task ##
                - Analyze the question and extract the most critical phrases or terms that capture the essence of the query.
                - If the question involves time-related options, calculate the latest time mentioned in the options and provide the search time range accordingly.
                - If the question does not involve time-related options, set the search time range to the question time minus 5 minutes.

                ## Keywords Extract Rules ##
                1. Keywords should be as short as possible and directly tied to the question's context, only focus on action and items.
                
                2. Use ONLY verb + direct noun (remove locations, adjectives, and extra details).

                3. If specific NAME in the question, the keywords MUST contain NAMES.
                
                ## Time Calculation Rules ##
                1. Time format is always HHMMSSFF.
                
                2. For time-related questions:
                - Identify the latest time mentioned in the options.
                - Set the search time range to that time BASED ON THE QUESTION TIME.
                
                
                3. For non-time-related questions:
                - Set the search time range to the question time minus 5 minutes.
                
                5. The search time range MUST ALWAYS PRECEDE the question time.
                
                6. Please decide the Search Time Range Carefully!!!!
                
                
                ## Output Format ##
                Keywords: [extracted_keyword_here]
                Search Time Range: [search_time_range]

                ## Examples ##
                1. Q: What did we end up eating the last time we ordered takeout?
                Options:
                A. "KFC",
                B. "Noodles",
                C. "Dumplings",
                D. "Pizza",
                Question Time: DAY1 18304214
                Output: 
                Keywords: [ordered takeout]
                Search Time Range: [DAY1 18250000]

                2. Q: When was the last time items were taken from the refrigerator?
                Options:
                A. "Last night",
                B. "Yesterday at noon",
                C. "This morning",
                D. "Yesterday morning",
                Question Time: DAY2 10443918
                Output: 
                Keywords: [take out from refrigerator]
                Search Time Range: [DAY2 09000000]

                3. Q: When was the last time I received a receipt?
                Options:
                A. "The day before yesterday afternoon",
                B. "The day before yesterday at noon",
                C. "Yesterday afternoon",
                D. "Last night",
                Question Time: DAY5 16353500
                Output:
                Keywords: [receive receipt]
                Search Time Range: [DAY4 23590000]

                4. Q: What food was the microwave last used to heat?
                Options:
                A. "Pizza",
                B. "Soup",
                C. "Rice",
                D. "Noodles",
                Question Time: DAY3 12004512
                Output:
                Keywords: [use microwave]
                Search Time Range: [DAY3 11550000]
                """
                
video_evidence_prompt ="""
                    Please analyze the provided video and help me determine if the provided information is helpful for answering the given question: {question}

                    - If the video segment is relevant, extract the relevant information from the video segment that helps answer the question. Return the response in the following JSON format:

                        \"I can provide evidence. Evidence: <state relevant information from the video that helps answer the question>\"

                    - If the video segment is not relevant, set the "status" value to "false" and return the following format:

                        \"I can't provide evidence.\"
                """
text_evidence_prompt ="""
                video caption: {video_caption}
                
                Format explanation:
                - The video caption follows this format:
                    "Video Time: It is DAY<video_date>, <video_start_time> to <video_end_time>. Video Content: <video_caption>"
                - Time format is HHMMSSFF (Hours, Minutes, Seconds, Frames).
                - The video content consists of captions describing events in the video from a first-person perspective.

                Please analyze the provided text, which consists of video captions describing events in the video clip from a first-person perspective.

                Help me determine if the provided information is useful for answering the given question: {question}

                If the text is relevant:
                - Extract the relevant information from the text that helps answer the question.
                - Return the response in the following format:

                    "I can provide evidence. Evidence: <state relevant information from the text>"

                If the text is not relevant:
                - Return the following format:

                    "I can't provide evidence."
                """
                
with_evidence = """You will receive a series of evidence cards. Each card provides information in the following format:
        - **time**: DAYX_HHMMSSFF_HHMMSSFF  
        - **evidence_info**: Extracted visual information about the event.  

        Where **time** records a period during which specific events or actions occurred. `DAYX` indicates the day on which the events happened. `HHMMSSFF` specifies the exact time range, where `HH` is hours(0-24), `MM` is minutes)0-60), `SS` is seconds(0-60), and `FF(0-20)`.

        **Evidence cards:**  
        {combined_string}

        Using the information provided in the evidence cards, analyze the events and actions to answer the following question:

        **Question:**  
        {question}

        **Question time:**  
        {question_time}

        **Options:**  
        {formatted_options}

        ### Instructions:
        - Use the evidence cards to determine the most appropriate answer.  
        - Provide your answer as the letter corresponding to the correct option from the given choices.  
        - To calculate time differences, assume that:
            - If the question refers to **"yesterday"**, the target time is one day before the **Question time**.
            - If the question refers to **"the day before yesterday"**, the target time is two days before the **Question time**.
            - If the question refers to **"the day before the last event"**, compute the difference between the **Question time** and the time of the last event.
            - Pay attention to the **DAYX** and calculate the correct day difference.
        - Respond with **the letter only** (e.g., `A`, `B`, `C`, or `D`)."""

without_evidence = """You will receive a series of **related events**, which provide contextual information relevant to the question. Each related event is structured as follows:
        - **time**: DAYX_HHMMSSFF_HHMMSSFF  
        - **event_info**: Descriptive information about the event.  

        Where **time** records a period during which specific events or actions occurred. `DAYX` indicates the day on which the events happened. `HHMMSSFF` specifies the exact time range, where `HH` is hours(0-24), `MM` is minutes)0-60), `SS` is seconds(0-60), and `FF(0-20)` is frames.

        **Related video events:**  
        {caption_results}

        Using the information provided in the related events, analyze the context and answer the following question:

        **Question:**  
        {question}

        **Question time:**  
        {question_time}

        **Options:**  
        {formatted_options}

        ### Instructions:
        - Refer to all the related events to determine the most appropriate answer.  
        - Base your reasoning on the provided information and logical inference.  
        - To calculate time differences, assume that:
            - If the question refers to **"yesterday"**, the target time is one day before the **Question time**.
            - If the question refers to **"the day before yesterday"**, the target time is two days before the **Question time**.
            - If the question refers to **"the day before the last event"**, compute the difference between the **Question time** and the time of the last event.
            - Pay attention to the **DAYX** and calculate the correct day difference.
        - Provide your answer as the letter corresponding to the correct option from the given choices.  
        - Respond with **the letter only** (e.g., `A`, `B`, `C`, or `D`)."""