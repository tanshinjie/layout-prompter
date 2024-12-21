import re 

def _extract_labels_and_bboxes_from_html(predition: str):
    if re.findall('<style>', predition):
            raise RuntimeError("Incorrect html detected")
    
    labels = re.findall('<div class="(.*?)"', predition)[1:]  # remove the canvas
    x = re.findall(r"left:.?(\d+)px", predition)[1:]
    y = re.findall(r"top:.?(\d+)px", predition)[1:]
    w = re.findall(r"width:.?(\d+)px", predition)[1:]
    h = re.findall(r"height:.?(\d+)px", predition)[1:]

    print(f"labels: {labels}")
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"w: {w}")
    print(f"h: {h}")

    if not (len(labels) == len(x) == len(y) == len(w) == len(h)):
        print("Error in _extract_labels_and_bboxes_from_html")
        return
        raise RuntimeError(
            "The number of labels, x, y, w, h are not the same "
            f"(#labels = {len(labels)}, #x = {len(x)}, #y = {len(y)}, #w = {len(w)}, #h = {len(h)})."
        )
    return True


# response.json
prediction = "```html\n<html>\n<body>\n  <!-- Canvas for the poster layout -->\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px; background-color: #f0f0f0;\">\n    \n    <!-- Underlay positioned to avoid overlapping with salient content -->\n    <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: rgba(200, 200, 200, 0.5); z-index: -1;\"></div>\n    \n    <!-- Text elements positioned above the underlay -->\n    <div class=\"text\" style=\"position: absolute; left: 5px; top: 7px; width: 47px; height: 13px; background-color: #fff;\">Text 0</div>\n    <div class=\"text\" style=\"position: absolute; left: 5px; top: 24px; width: 73px; height: 14px; background-color: #fff;\">Text 1</div>"
prediction = "```html\n<html>\n<body>\n  <!-- Canvas for the poster layout -->\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px; background-color: lightgray;\">\n    \n    <!-- Underlay background positioned to avoid salient content -->\n    <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: lightblue; z-index: -1;\"></div>\n    \n    <!-- Text element 0 positioned at the top section -->\n    <div class=\"text\" style=\"position: absolute; left: 5px; top: 7px; width: 47px; height: 13px; background-color: white; z-index: 1;\">\n      Text 0\n    </div>\n    \n    <!-- Text element 1 positioned below text element 0 -->\n    <div class=\"text\" style=\"position: absolute; left: 5px; top: 24px; width: 73px; height: 14px; background-color: white; z-index: 1;\">\n      Text 1\n    </div>\n  </div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n  <!-- Canvas for the poster layout -->\n  <div class=\"canvas\" style=\"position: absolute; left: 0px; top: 0px; width: 102px; height: 150px; background-color: #f0f0f0;\"></div>\n  \n  <!-- Underlay element positioned to align with the content constraint while avoiding salient content -->\n  <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: #dcdcdc;\"></div>\n  \n  <!-- Text element 0 positioned to avoid overlap with the underlay and other important content -->\n  <div class=\"text\" style=\"position: absolute; index: 0; left: 5px; top: 7px; width: 47px; height: 13px; background-color: #ffffff;\">Text 0</div>\n  \n  <!-- Text element 1 positioned similarly to text element 0 for consistency, while avoiding salient content -->\n  <div class=\"text\" style=\"position: absolute; index: 1; left: 5px; top: 24px; width: 73px; height: 14px; background-color: #ffffff;\">Text 1</div>\n</body>\n</html>\n```"
prediction = "Here's a potential layout for the given constraints:"
prediction = "```html\n<html>\n<body>\n  <!-- Canvas for the poster layout -->\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px; background-color: #f0f0f0;\"></div>\n  \n  <!-- Underlay to serve as background for text or logo -->\n  <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: #e0e0e0;\"></div>\n  \n  <!-- First text element, positioned to avoid salient content -->\n  <div class=\"text\" style=\"position: absolute; left: 5px; top: 7px; width: 47px; height: 13px; background-color: #c0c0c0;\"></div>\n  \n  <!-- Second text element, positioned to avoid salient content -->\n  <div class=\"text\" style=\"position: absolute; left: 5px; top: 24px; width: 73px; height: 14px; background-color: #c0c0c0;\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px; background-color: lightgray;\"></div>\n  \n  <!-- Underlay for text or logo -->\n  <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: white; opacity: 0.5;\"></div>\n  \n  <!-- Text elements, avoiding overlap with salient content -->\n  <div class=\"text\" style=\"position: absolute; left: 10px; top: 10px; width: 47px; height: 13px; background-color: none;\">\n    Text 0\n  </div>\n  <div class=\"text\" style=\"position: absolute; left: 10px; top: 30px; width: 73px; height: 14px; background-color: none;\">\n    Text 1\n  </div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px; background-color: #f0f0f0;\"></div>\n  \n  <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: #dcdcdc;\"></div>\n  \n  <div class=\"text\" style=\"position: absolute; left: 5px; top: 7px; width: 47px; height: 13px; background-color: #ffffff;\">\n    <span style=\"font-size: 10px;\">Text Block 0</span>\n  </div>\n  \n  <div class=\"text\" style=\"position: absolute; left: 5px; top: 24px; width: 73px; height: 14px; background-color: #ffffff;\">\n    <span style=\"font-size: 10px;\">Text Block 1</span>\n  </div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px;\">\n    <!-- Underlay background for text or logo -->\n    <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: #f0f0f0;\"></div>\n    \n    <!-- Text elements positioned to avoid salient content and overlaid on underlay -->\n    <div class=\"text\" style=\"position: absolute; left: 10px; top: 72px; width: 47px; height: 13px; background: transparent;\">Text 0</div>\n    <div class=\"text\" style=\"position: absolute; left: 10px; top: 90px; width: 73px; height: 14px; background: transparent;\">Text 1</div>\n  </div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px; background-color: lightgrey;\"></div>\n  \n  <!-- Underlay Element -->\n  <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px; background-color: rgba(200, 200, 200, 0.5);\"></div>\n  \n  <!-- Text Elements -->\n  <div class=\"text\" style=\"position: absolute; left: 5px; top: 7px; width: 47px; height: 13px; background-color: white; z-index: 1;\">Text 0</div>\n  \n  <div class=\"text\" style=\"position: absolute; left: 5px; top: 24px; width: 73px; height: 14px; background-color: white; z-index: 1;\">Text 1</div>\n</body>\n</html>\n```"

# runtime_error_label_not_same_response.json
prediction = "```html\n<html>\n<body>\n<div class=\"canvas\" style=\"left: 0px; top: 0px; width: 102px; height: 150px\"></div>\n<div class=\"text\" style=\"index: 0; left: 10px; top: 10px; width: 82px; height: 12px\"></div>\n<div class=\"text\" style=\"index: 1; left: 10px; top: 30px; width: 82px; height: 10px\"></div>\n<div class=\"underlay\" style=\"index: 2; left: 8px; top: 68px; width: 85px; height: 78px\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n<div class=\"canvas\" style=\"left: 0px; top: 0px; width: 102px; height: 150px\"></div>\n<div class=\"text\" style=\"index: 0; left: 10px; top: 10px; width: 82px; height: 15px\"></div>\n<div class=\"text\" style=\"index: 1; left: 12px; top: 30px; width: 78px; height: 20px\"></div>\n<div class=\"underlay\" style=\"index: 2; left: 8px; top: 68px; width: 85px; height: 78px\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n<div class=\"canvas\" style=\"left: 0px; top: 0px; width: 102px; height: 150px\"></div>\n<div class=\"text\" style=\"index: 0; left: 10px; top: 10px; width: 82px; height: 15px\"></div>\n<div class=\"text\" style=\"index: 1; left: 12px; top: 30px; width: 78px; height: 14px\"></div>\n<div class=\"underlay\" style=\"index: 2; left: 8px; top: 68px; width: 85px; height: 78px\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n<div class=\"canvas\" style=\"left: 0px; top: 0px; width: 102px; height: 150px\"></div>\n<div class=\"text\" style=\"index: 0; left: 10px; top: 10px; width: 82px; height: 15px\"></div>\n<div class=\"text\" style=\"index: 1; left: 10px; top: 30px; width: 82px; height: 15px\"></div>\n<div class=\"underlay\" style=\"index: 2; left: 8px; top: 66px; width: 85px; height: 80px\"></div>\n</body>\n</html>\n```\n"
prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"position: absolute; left: 0px; top: 0px; width: 102px; height: 150px;\"></div>\n  \n  <!-- Text elements positioned to avoid salient content -->\n  <div class=\"text\" style=\"position: absolute; left: 10px; top: 10px; width: 82px; height: 12px;\"></div>\n  <div class=\"text\" style=\"position: absolute; left: 10px; top: 30px; width: 82px; height: 12px;\"></div>\n  \n  <!-- Underlay placed under text elements to ensure readability -->\n  <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 68px; width: 85px; height: 78px;\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n<div class=\"canvas\" style=\"left: 0px; top: 0px; width: 102px; height: 150px\"></div>\n<div class=\"text\" style=\"index: 0; left: 10px; top: 10px; width: 82px; height: 15px\"></div>\n<div class=\"text\" style=\"index: 1; left: 10px; top: 30px; width: 82px; height: 15px\"></div>\n<div class=\"underlay\" style=\"index: 2; left: 8px; top: 68px; width: 85px; height: 78px\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n<div class=\"canvas\" style=\"left: 0px; top: 0px; width: 102px; height: 150px\"></div>\n<div class=\"text\" style=\"index: 0; left: 10px; top: 10px; width: 82px; height: 12px\"></div>\n<div class=\"text\" style=\"index: 1; left: 15px; top: 30px; width: 72px; height: 10px\"></div>\n<div class=\"underlay\" style=\"index: 2; left: 5px; top: 65px; width: 92px; height: 80px\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"left: 0px; top: 0px; width: 102px; height: 150px\"></div>\n  <div class=\"text\" style=\"index: 0; left: 10px; top: 10px; width: 82px; height: 12px\"></div>\n  <div class=\"text\" style=\"index: 1; left: 15px; top: 30px; width: 72px; height: 18px\"></div>\n  <div class=\"underlay\" style=\"index: 2; left: 8px; top: 66px; width: 85px; height: 78px\"></div>\n</body>\n</html>\n```"
prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"position: relative; width: 102px; height: 150px;\"></div>\n  <div class=\"text\" style=\"position: absolute; left: 5px; top: 10px; width: 92px; height: 14px;\"></div>\n  <div class=\"text\" style=\"position: absolute; left: 10px; top: 30px; width: 82px; height: 12px;\"></div>\n  <div class=\"underlay\" style=\"position: absolute; left: 8px; top: 66px; width: 86px; height: 80px;\"></div>\n</body>\n</html>\n```" # error
# prediction = "```html\n<html>\n<body>\n  <div class=\"canvas\" style=\"position: absolute; left: 0px; top: 0px; width: 102px; height: 150px; background-color: #f0f0f0;\"></div>\n  \n  <!-- Text elements aligned at the top -->\n  <div class=\"text\" style=\"position: absolute; left: 10px; top: 10px; width: 82px; height: 15px; background-color: #ffffff;\">\n    Text 0\n  </div>\n  <div class=\"text\" style=\"position: absolute; left: 10px; top: 30px; width: 82px; height: 15px; background-color: #ffffff;\">\n    Text 1\n  </div>"
prediction = "```html\n<!DOCTYPE html>\n<html>\n<head>\n    <style>\n        .canvas {\n            position: relative;\n            width: 102px;\n            height: 150px;\n            background-color: #f0f0f0;\n        }\n        .logo {\n            position: absolute;\n            width: 27px;\n            height: 55px;\n            background-color: #d1d1d1;\n            left: 2px;\n            top: 79px;\n        }\n        .text-1 {\n            position: absolute;\n            width: 67px;\n            height: 12px;\n            background-color: #e1e1e1;\n            left: 34px;\n            top: 83px;\n        }\n        .text-2 {\n            position: absolute;\n            width: 67px;\n            height: 12px;\n            background-color: #e1e1e1;\n            left: 34px;\n            top: 95px;\n        }\n        .text-3 {\n            position: absolute;\n            width: 67px;\n            height: 12px;\n            background-color: #e1e1e1;\n            left: 34px;\n            top: 107px;\n        }\n        .underlay {\n            position: absolute;\n            width: 102px;\n            height: 13px;\n            background-color: #c1c1c1;\n            left: 0px;\n            top: 120px;\n        }\n    </style>\n</head>\n<body>\n    <div class=\"canvas\">\n        <div class=\"logo\"></div>\n        <div class=\"text-1\"></div>\n        <div class=\"text-2\"></div>\n        <div class=\"text-3\"></div>\n        <div class=\"underlay\"></div>\n    </div>\n</body>\n</html>\n```" # error
prediction = "```html\n<html>\n<body>\n<style>\n  .canvas {\n    position: absolute;\n    width: 102px;\n    height: 150px;\n    background-color: #f0f0f0; /* Background color for the canvas */\n  }\n  .logo {\n    position: absolute;\n    left: 2px;\n    top: 79px;\n    width: 27px;\n    height: 55px;\n    background-color: #d3d3d3; /* Placeholder color for logo */\n  }\n  .text {\n    position: absolute;\n    background-color: rgba(255, 255, 255, 0.8); /* Background for text to enhance readability */\n  }\n  .underlay {\n    position: absolute;\n    background-color: rgba(200, 200, 200, 0.5); /* Underlay background color */\n  }\n</style>" # error



_extract_labels_and_bboxes_from_html(predition=prediction)