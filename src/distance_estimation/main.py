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

    calc.gcc_phat(music_type=music)
    
    output_path = f'./../../data/distance_estimation/music{music}_mono/csv_files'       # 保存するCSVファイルのパス

    visualize.plot_embedding_error(
        music_type=music,
        delay_time_errors_file=f'{output_path}/delay_time_errors.csv',
        pesq_scores_file=f'{output_path}/pesq_scores.csv',
        embedding_amplitudes=np.linspace(0, 1, 25),
    )

print("Completed!!!")
