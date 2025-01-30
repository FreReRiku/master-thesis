.PHONY: all room_simulation distance_estimation

# デフォルトターゲット
all: room_simulation distance_estimation

# src/room_simulation/main.py を実行
room_simulation:
	cd src/room_simulation && python main.py

# src/distance_estimation/main.py を実行
distance_estimation:
	cd src/distance_estimation && python main.py
