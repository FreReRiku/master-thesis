"""
main.py
----------

Created by FreReRiku on 2024/12/30
"""

import calc
import numpy as np
import visualize

music_type = [1, 2]         # 使用する楽曲のタイプ


for music in music_type:

    calc.gcc_phat(music)
    
    visualize.plot_embedding_error(music)
    visualize.plot_impulse(music)
    visualize.plot_mean_embedded_csp(music)

print("Completed!!!")
