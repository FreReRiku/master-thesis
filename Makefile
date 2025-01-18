.PHONY: all create_dirs room_simulation distance_estimation

# デフォルトターゲット
all: create_dirs room_simulation distance_estimation

# 必要なディレクトリを作成
create_dirs:
	@echo "Creating necessary directories..."
	@root_dir="." && \
	dirs=(\
	"$$root_dir/data/distance_estimation/music1_mono/csv_files" \
	"$$root_dir/data/distance_estimation/music2_mono/csv_files" \
	"$$root_dir/data/room_simulation/room_info" \
	"$$root_dir/data/room_simulation/sr_and_spk" \
	"$$root_dir/figure/distance_estimation/archive" \
	"$$root_dir/figure/distance_estimation/music1_mono/plot_csp" \
	"$$root_dir/figure/distance_estimation/music1_mono/plot_embedding_error" \
	"$$root_dir/figure/distance_estimation/music1_mono/plot_impulse" \
	"$$root_dir/figure/distance_estimation/music2_mono/plot_csp" \
	"$$root_dir/figure/distance_estimation/music2_mono/plot_embedding_error" \
	"$$root_dir/figure/distance_estimation/music2_mono/plot_impulse" \
	"$$root_dir/figure/room_simulation" \
	"$$root_dir/sound/distance_estimation/music1_mono" \
	"$$root_dir/sound/distance_estimation/music2_mono" \
	"$$root_dir/sound/original" \
	"$$root_dir/sound/room_simulation" \
	) && \
	for dir in $${dirs[@]}; do \
		mkdir -p "$$dir"; \
	done

# src/room_simulation/main.py を実行
room_simulation:
	cd src/room_simulation && python main.py

# src/distance_estimation/main.py を実行
distance_estimation:
	cd src/distance_estimation && python main.py
