import random
import os
from shutil import copy
import glob
import numpy as np

# define params
# [name, folder, postfix]
methods = [["DirectPaste", "paste_ours", "-direct_paste.png"],
          ["NeuralIllu", "neural_illu", "_place_blend.png"],
          ["UCSD", "ucsd", "_place_blend.png"],
          ["DoveNet", "dovenet", "-direct_paste.png"],
          ["Ours", "paste_ours", "-out_rgb.png"]]
background = ["Background", "paste_ours", "-scene_rgb.png"]
picked = [
"6114_15282_0",
"18680_100612_0",
"5976_10152_0",
"8100_227_0",
"11604_102274_0",
"5727_102274_0",
"2471_35876_0",
"1143_101000_0",
"9720_35584_0",
"11072_35296_0",
"9621_17610_0",
"33630_35447_0",
"141_12177_0",
"116_791_0",
"7963_59541_0",
"18659_16782_0",
"33753_59541_0",
"12058_33339_0",
"9775_105213_0",
"1934_55148_0",
]

img_lists = glob.glob("./paste_ours/*direct_paste.png")

f = open('./compare.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 15
for p in picked:
    i = os.path.basename(p).split('-')[0]

    img = f"./paste_ours/{i}-scene_rgb.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">RGB {i}</p>"
    msg += tmp

    img = f"./paste_ours/{i}-direct_paste.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Direct paste</p>"
    msg += tmp

    img = f"./dovenet/{i}-direct_paste.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">DoveNet</p>"
    msg += tmp

    img = f"./ucsd/{i}_place_blend.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">UCSD</p>"
    msg += tmp

    img = f"./neural_illu/{i}_place_blend.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Neural illum</p>"
    msg += tmp

    img = f"./paste_ours/{i}-out_rgb.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Geometry attention (ours)</p>"
    msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()

figw = 20
N_USERS = 1
for user_num in range(N_USERS):

    # html_path = './user_study/user_{}/'.format(user_num)

    # if not os.path.exists(html_path):
    #     os.makedirs(html_path, exist_ok=True)

    # html_path += 'index.html'


    html_content = '<!DOCTYPE html><html>'
    html_content += '''<style>  
        .container {  
            width: 80%;  
            margin: auto;  
            display: flex;  
            justify-content: center;  
            align-items: center;  
            }    

        .container_num {  
            display: flex;  
            justify-content: center;  
            align-items: center;
            width: 16%;  
            float: left;
            }    

        .container_img {  
            display: flex;  
            justify-content: center;  
            align-items: center;
            width: 33.3%;  
            float: left;
            }

        .element{
          width: 100%;
          height: 100%;
          position: relative;
          top: 90%; left: 0%;
          transform: translate(50%, 0%);
        }

        #checkbox{
        position: relative;
          top: 90%; left: 0%;
          transform: translate(0%, 50%);
          }

        .helper {
            display: inline-block;
            height: 100%;
            vertical-align: middle;
        }

        .column_num {
          float: left;
          width: 15.66%;
          padding: 5px;
        }

        .column {
          float: left;
          width: 18.5%;
          padding: 5px;
        }

        .column_checkbox {
          float: left;
          width: 17.5%;
          padding: 5px;
        }

        /* Clear floats after image containers */
        .row::after {
          content: "";
          clear: both;
          display: table;
        }
        </style>  '''

    html_content += '''<h2 style="margin-top: 5%; font-size: 22pt;">Instructions</h2>'''
    html_content += '''<p style ="font-size:16pt" >Two methods and the background are shown for each scene, where an object is placed at a location in background. Pick whichever method you think best represents the illumination in the scene, or whether there is no difference. If you think neither is appropriate, please select similar.</p>'''

    html_content += '''<form action="https://mailthis.to/yimingq@sfu.ca"
               method="POST" encType="multipart/form-data">'''
    html_content += '''Name: <input type="text" name="usrname">'''

    count = 1
    for k,p in enumerate(img_lists):

        prefix = os.path.basename(p).split('-')[0]
        if prefix not in picked:
            continue
        n = len(methods)
        for i in range(n):
            for j in range(i + 1, n):
                html_content += '<h2 style="margin-top: 5%; font-size: 22pt;">Question {}</h2>'.format(count)
                count += 1

                html_content += '''
                                <div class="row">   
                                  <div class="column">
                                    <p style="text-align:center">Method 1</p>
                                  </div>
                                  <div class="column">
                                    <p style="text-align:center">Background</p>
                                  </div>
                                   <div class="column">
                                    <p style="text-align:center">Method 2</p>
                                  </div>
                                </div>'''
                if random.random()>0.5:
                    mid_1, mid_2 = i, j
                else:
                    mid_1, mid_2 = j, i
                img_path_1 = f"./{methods[mid_1][1]}/{prefix}{methods[mid_1][2]}"
                img_path_2 = f"./{methods[mid_2][1]}/{prefix}{methods[mid_2][2]}"
                img_path_gt = f"./{background[1]}/{prefix}{background[2]}"

                html_content += '''<div class="row">
                              <div class="column">
                                <img src="{}" style="width:100%">
                              </div>
                              <div class="column">
                                <img src="{}" style="width:100%">
                              </div>
                              <div class="column">
                                <img src="{}" style="width:100%">
                              </div>
                            </div>'''.format(img_path_1, img_path_gt, img_path_2)

                html_content += '''
                                <div class="row">   
                                  <div class="column">
                                    <p style="text-align:center"><input type="checkbox" name="{}" id="{}"><label>1 Is Better</label></p>
                                  </div>
                                  <div class="column">
                                    <p style="text-align:center"><input type="checkbox" name="{}" id="{}"><label>Similar</label></p>
                                  </div>
                                   <div class="column">
                                    <p style="text-align:center"><input type="checkbox" name="{}" id="{}"><label>2 Is Better</label></p>
                                  </div>
                                </div>'''.format(methods[mid_1][0] + "&@VS&@" + methods[mid_2][0] + "&@" + methods[mid_1][0], methods[mid_1][0] + "&@VS&@" + methods[mid_2][0] + "&@" + methods[mid_1][0],
                                                 methods[mid_1][0] + "&@VS&@" + methods[mid_2][0] + "&@" + "Similar", methods[mid_1][0] + "&@VS&@" + methods[mid_2][0] + "&@" + "Similar",
                                                 methods[mid_1][0] + "&@VS&@" + methods[mid_2][0] + "&@" + methods[mid_2][0], methods[mid_1][0] + "&@VS&@" + methods[mid_2][0] + "&@" + methods[mid_2][0])



    html_content += '''<div class='container'><input type="submit" value="Send" style="height:100px; width:400px;"></div></form>'''
    html_content += '</html>'

    with open(f"index_{user_num}.html", 'w') as f:
        f.write(html_content)