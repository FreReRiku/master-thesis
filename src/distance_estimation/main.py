"""
main.py
----------

Created by FreReRiku on 2024/12/30
"""

import calc
import visualize

music_type = [1, 2]         # 使用する楽曲のタイプ
emb_type = ["amplitude_modulation", "phase_modulation"]

for music in music_type:

    for emb in emb_type:

        calc.gcc_phat(music, emb)
        visualize.plot_embedding_error(music, emb)
        visualize.plot_impulse(music, emb)
        visualize.plot_mean_embedded_csp(music, emb)
        visualize.plot_embedded_frequencies(music, emb)
        visualize.plot_audio_waveform(music, emb)

print("Completed!!!")
