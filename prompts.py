import random

prompts_b = {
"long_thoughts_v2": """Your answer must contain 6 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture with given popular tags, or descriptions, or your memories for each characters to determine who is who.
# 2. Key details
Here you need to determine key details on comic and list them.
# 3. Long description
Here come up with a long and detailed description of image content. Be creative, mention all detailes you listed above and other important things.
# 4. Detailed description for each character
## Name 1
Detailed and long description for the first character
## Name 2
Same for each one (if present)
</format>
""",
"long_thoughts": """Your answer must contain 6 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.
If no characters are listed in input - just write here "No named characters"
# 2. General description
A one-two paragraph summary of the image. Mention all individual parts/objects/characters/positions/interactions/etc.
# 3. Detailed description for each character
## Character name 1 (put here the name if any)
In very detail write about features, poses, look, used objects, interactions, and other things for character on the picture.
## Character name 2 (put here the name if any)
Same for each character.
...
# 4. Individual Parts
List the individual things you see in the image and their relative positions to other parts. Use a numbered list of between 5 and 20 items depending on image complexity.
# 5. Texts on image
Mention every texts that you notice on image, including types (a speech bubble, watermark, banner, etc.) and content.
# 6. Background and effects
Give some info about objects on background, describe the location (if seen). Then mention effects (style, camera angle, clarity/blurrines, effects like depth of field, strange angle/forshortening, etc.)
</format>
""",
"json": """Use json-style caption for given image with following structure:
{"character" : "Description for character or object. Name (if defined), main details, features, position, pose, etc.",
/or in case of multiple
"character_1" : "Description for first"
"character_2" : "Description for second ",
"character_N"...
/or if there are no characters
"main content" : "long and detailed description of main content of image that might be the main focus if characters are missing",
/
"background" : "Detailed descritpion of background and it's content",
"image_effects" : "If there are some visual effects like fisheye distortion, chromatic aberration, glitches, messy drawing or anything else - write about it. If it's just a general anime art - omit this field."
"texts" : "Speech bubbles, bars, marks, signs etc. with texts if present, else None",
"atmosphere" : "...",
}
In special cases you can add extra keys.
""",
"long": """Make a caption for given image with natural text. Use 2 to 5 paragraphs. Make your description long and vivid, mentioning all the details.
""",
"min_structured_md": """Your answer must contain 3 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.
If no characters are listed in input - just write here "No named characters"
# 2. Key details
Here you need to write about the key details on image, prefere using regular text.
# 3. Structured description
## General
Write about general composition, content of image, background and all things that are not related to characters directly.
## Character name 1 (put here the name if any)
Write about datails and content related to specific character, including features, poses, look, used objects, interactions, and other things.
## Character name 2 (put here the name if any)
Same for each character.
## Image effects
Mention image effect, style, camera angle
</format>
In general stick to shorter descriptions.
""",
"json_comic": """Use json-style caption to describe to comin, stick to following structure:
{
"comic_format": "menation the format, for example Comic of N frames",
"1st_frame": "Main description of the content for fist frame",
"2nd_frame": "Same for the second",
...
"Nth_ftame": "...",
"character_1": "Describe the characters in comic",
...
"character_N": "Separate description for each",
"meaning": "Try to guess general mood, vibe and meaning of the comic"
}
""",
"md_comic": """Use markdown format to describe to comic, 5 parts are recommended:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture with given popular tags, or descriptions, or your memories for each characters to determine who is who.
# 2. Key details
Here you need to determine key details on comic and list them.
# 3. Comic format
In this section come up with the description of comic format, how many pages there are, horisontal/vertical orientation and other things. Optionally you can list main characters here.
# 4. Details for each frame
## 4.1 Frame 1 (position)
Description for each frame, includding characters, objects, interactions, texts/speech bubbles and other things. Be detailed but not overdoo.
## 4.2 Frame 2 (position)
Same for each frame.
...
# 5. Extra comment
Here you should write general desciption and some other info about the image.
</format>
""",
"min_structured_json": """
Use json-style caption for given image with following structure:
{"General" : "Here you need to come up with general/common information about picture, overall composition. Stick to shorter phrases and tags instead of long purple prose. Avoid bullets and markdown, write in plain text.",
"character_1 (put here the name if any)" : "Description of first character."
"character_2 (if present" : "Description for second ",
"character_N"
...
"image_effects" : "Mention here effects on image if there are any distinct."
"texts" : "Speech bubbles, bars, marks, signs etc. with texts if present, else None",
"watermarks" : "If present",
}
Prefere shorter description and tags.
""",
"chroma-style": """Your task is to describe the picture in very detail using a structure of 4 parts.
### 1. Regular Summary:
[A one-paragraph summary of the image. The paragraph should mention all individual parts/things/characters/etc.]
### 2. Individual Parts:
[List the individual things you see in the image and their relative positions to other parts. Use a numbered list of between 5 and 30 items depending on image complexity.]
### 3. Midjourney-Style Summary:
[A summary that has higher concept density by using comma-separated partial sentences instead of proper sentence structure.]
### 4. DeviantArt Commission Request
[Write a description as if you're commissioning this *exact* image via someone who is currently taking requests.]
""",
"short":"""The caption for image should be quite short without long purple prose and slop. Cover main objects and details.
""",
}

prompts_names_only = {
    "long_thoughts_v2":True,
    "long_thoughts": True,
    "json": False,
    "long": False,
    "json_comic": False,
    "md_comic": True,
    "min_structured_md": True,
    "min_structured_json": False,
    "chroma-style": False,
    "short":False,
    }


def make_user_query(item, c_type, use_names, add_tags, add_characters, add_char_tags, add_description, underscores_replace = False):
    tags = item.get('tags', [])
    random.shuffle(tags)
    if underscores_replace:
        tags = [a.replace('_', ' ') if len(a)>3 else a for a in tags]
        tags_string = ', '.join(tags)
    else:
        tags_string = ' '.join(tags)
    
    user_request = '# Captioning format:\n'
    user_request += prompts_b[c_type]
    user_request += '\n'
    
    if add_tags:
        user_request += f"# Booru tags for the image\n[{tags_string}]\n\n"
    
    chars_tags = item.get('characters',[])

    if use_names:
        has_character_grounding = bool(chars_tags) or add_characters or add_char_tags or add_description
        if has_character_grounding:
            if underscores_replace:
                chars_tags = [a.replace('_', ' ') for a in chars_tags]
                chars_string = ', '.join(chars_tags)
            else:
                chars_string = ', '.join(chars_tags)
            
            if chars_string:
                user_request += f"# Characters on picture:\nHere are names/tags for characters from the picture, make sure to use them: [{chars_string}].\n\n"
            
            chars_popular_tags = (item.get('char_p_tags',"{'chars':{},'skins':{}}"))
            chars_description = (item.get('char_descr',"{'chars':{},'skins':{}}"))
            
            has_tags = len(chars_popular_tags['chars']) > 0 or len(chars_popular_tags['skins']) > 0
            has_descriptions = len(chars_description['chars']) > 0 or len(chars_description['skins']) > 0
            if (add_char_tags and has_tags) or (add_description and has_descriptions):
                
                user_request += "# Known traits for characters\n"
                user_request += (
                    "Use the following character traits as authoritative grounding for the named characters. "
                    "When describing each named character, include these traits and do not replace them with "
                    "unrelated visual traits from another identity. If the image appears to show conflicting "
                    "hair, eyes, clothing, accessories, or outfit details for a named character, prefer the "
                    "provided traits below over the conflicting visual evidence. This is especially important "
                    "for clothing and accessories: describe the named character wearing the outfit given in "
                    "their tags/descriptions instead of copying the outfit from the source image.\n"
                )
                char_underscores = underscores_replace
                
                if add_char_tags and has_tags:
                    user_request += "Here are popular tags for each characters on picture:\n"
                    
                    for c_name, c_tags in chars_popular_tags['chars'].items():
                        name = c_name.replace('_',' ') if char_underscores else c_name
                        tags_s = (', '.join([a.replace('_', ' ') if len(a)>3 else a for a in c_tags]) if char_underscores else
                                ', '.join(c_tags))
                        user_request += f"{name}: [{tags_s}]\n"
                    if len(chars_popular_tags['skins']) > 0:
                        user_request += "Extra tags for characters skins:\n"
                        for c_name, c_tags in chars_popular_tags['skins'].items():
                            name = c_name.replace('_',' ') if char_underscores else c_name
                            tags_s = (', '.join([a.replace('_', ' ') if len(a)>3 else a for a in c_tags]) if char_underscores else
                                    ', '.join(c_tags))
                            user_request += f"{name}: [{tags_s}]\n"
                        
                if add_description and has_descriptions:

                    user_request += "Here are general descriptions for each characters on the picture:\n"
                    for c_name, c_descr in chars_description['chars'].items():
                        name = c_name.replace('_',' ') if char_underscores else c_name
                        user_request += f"## {name}\n{c_descr}\n\n"
                    if len(chars_description['skins']) > 0:
                        user_request += "Here are also descriptions for specific skin of characters:\n"
                        for c_name, c_descr in chars_description['skins'].items():
                            name = c_name.replace('_',' ') if char_underscores else c_name
                            user_request += f"## {name}\n{c_descr}\n\n"
        else:
            user_request += "# Characters on picture:\nTry to recognize the characters in the picture and use their names.\n"
            
        user_request += '\n'
    else:
        user_request += "# Characters on picture:\nAvoid to guess names for characters.\n"
    
    return user_request
        
system_prompt = "You are image captioning expert. Describe user's picture according to requested format and instructions."
