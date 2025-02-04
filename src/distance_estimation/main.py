"""
main.py
----------

Created by FreReRiku on 2024/12/30
"""

import calc
import visualize

music_type = [1, 2]         # 使用する楽曲のタイプ
emb_type = ["amplitude_modulation", "phase_modulation"]
variable_parameters = ["embedding_ratio", "embedding_freq_ratio", "num_embedding_frames"]

for music in music_type:

    for emb in emb_type:
        
        for variable in variable_parameters:

            # 遅延推定の計算
            print(f'music_type{music}の{emb}処理を行います.')
            print(f'{variable}をループさせて計算を行います.')
            calc.gcc_phat(music, emb, variable)

            # 可視化処理
            visualize.AM_vs_PM(music, emb, variable)
            visualize.plot_embedding_error(music, emb, variable)
            visualize.plot_impulse(music, emb, variable)
            visualize.plot_mean_embedded_csp(music, emb, variable)
            visualize.plot_mean_csp1(music, emb, variable)
            visualize.plot_mean_csp2(music, emb, variable)
            visualize.plot_mean_csp_ws(music, emb, variable)
            visualize.plot_embedded_frequencies(music, emb, variable)
            visualize.plot_audio_waveform(music, emb, variable)

print("Completed!!!")
