"""
main.py
----------

Created by FreReRiku on 2024/12/29
"""

import simulate

music_type = [1, 2]

for music in music_type:
    print(f'music_type{music}の音響シミュレーションを行います')
    simulate.room(music)
