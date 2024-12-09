# EXPLANATION_OF_room_simulation.py

## wavファイルの読み込み

1. channelをリスト型として定義します。
2. seedsの数([1, 1234]なので2回)だけ、forループでwavfileを読み込みます。

スピーカーの数だけseedを設定していることになっており、今回は2個用意されているので、その2個のスピーカーに対して音源を再生できるようにするためにwavファイルを読み込みます。

【要検討】seedを使う場面ある...? いらないんだったらチャンネル数を
```python
channels = 2

for i in range(channels)
```

とかにした方がいいような気がするのだが...

```python:room_simulation.py
wavfile.read
```

wavfile.read関数は(音源そのものとサンプリング周波数)をタプルとして返します。アンパックされるため、値を受け取る変数を2個用意する必要があります。

## 部屋の設定

### 残響時間と部屋の寸法

```python:room_simulation.py
rt60
```

Reverberation Time 60dBを意味します。
音源が停止した後、音圧レベルが60dB減少するまでの時間を指します。

```python:room_simulation.py
room_dimentions
```

部屋の寸法をここに入力します。二次元にすると平面空間として扱われます。

### Sabineの残響式

Sabine（セービン/サービン）の式は以下のように定義されます。

${T=KVA}$

${T}$：残響時間[s]

${K}$：定数（常温時は0.162）

${V}$：室容積[$m^3$]

${A}$：吸音力（= 室表面積 × 吸音率）

```python:room_simulation.py
pra.inverse_savine(rt60, room_dimensions)
```

この、pyroomacousticsのinverse_savine関数ではsabineの公式を逆にして、ISMシミュレーターのパラメーターを求めています。（ISM: Image Source Method）

#### Image Source Method について

イメージソース法（Image Source Method）とは、音響シミュレーションで音の反射を計算するための技術の一つです。この方法は、特に室内の音響シミュレーションにおいて、音源から発せられた音が、壁や天井、床等で反射する様子を計算するのに適しています。

### 壁の材質を決める

以下のように設定しています。

```python:room_simulation.py
m = pra.make_materials(
    ceiling =   "plasterboard",
    floor   =   "carpet_cotton",
    east    =   "plasterboard",
    south   =   "plasterboard",
    west    =   "plasterboard",
    north   =   "plasterboard",
)
```

### 部屋をつくる・マイクを設置する

以下のようにして部屋を作ります。

```python:room_simulation.py
room = pra.ShoeBox(
    room_dimensions,
    fs          =   fs,
    materials   =   m, # 変更前: pra.Material(e_absorption)
    max_order   =   max_order # 変更前: 5
)
```

マイクも変数mic_locにベクトル（座標）として設置します。
上記で設定した変数roomにマイクを追加していきます。

```python:room_simulation.py
mic_loc = [1.75, 1.75, 1.6]
room.add_microphone(mic_loc)
```
add_microphone(mic_loc) はself.add(MicrophoneArray(loc, fs, directivity))を返します。

### マイク設置

指定した変数mic_locの座標の場所にマイクを設置します。

### 音源設置

音源を以下のように設定します。

```python:room_simulation.py
room.add_source([3.4, 0.5, 0.5], signal=channels[0])
room.add_source([3.4, 2.3, 0.5], signal=channels[1])
```

### 部屋表示

```python:room_simulation.py
room.plot()
```

上記の関数において、画像ソースが描画されていない場合、バウンディングボックスを使用してプロットの限界を正しく設定する必要があります。

- 上記の関数は、(fig, ax)のタプルを返します。

```python:room_simulation.py
ax.set_xlim([0, 3.6])
ax.set_ylim([0, 3.6])
ax.set_zlim([0, 2.5])
```

上記のコードは表示する尺の長さを設定します。

## シミュレーション&保存

### インパルス応答を計算

---

そもそもインパルス応答とは...?

インパルス応答とは、あるシステムにインパルス（時間的に継続時間が非常に短い信号）を入力した場合の、システムの出力のことです。

---

以下のコードでインパルス応答を計算します。

```python:room_simulation.py
room.compute_rir()
```

すべての音源とマイク間の室内インパルス応答を計算します。


### インパルス応答を保存

コード内の変数iはマイクのループカウンター、jはスピーカーのループカウンターを表しています。

計算によって得られたインパルス応答を

```python:room_simulation.py
/sound_data/room_simulation/
```

に保存します。この際、マイクは1つ、スピーカーは2つあるため、2つのインパルス応答の結果がwavファイルとして保存されます。

#### sf.write関数の説明

```python:room_simulation.py
sf.write(file, data, samplerate)
```

となっています。

#### シミュレーションについて

```python:room_simulation.py
room.simulate(return_premix=True)
```

上記の関数を用いることによってシミュレーションを行うことができます。

引数内にあるreturn_premix=Trueについて、"True" に設定すると、この関数は (n_sources, n_mics, n_samples) 形式の配列を返す仕組みになっています。

### 各音源のみを再生した音を保存する

ここでは、i、つまり、スピーカーの数だけ音源が保存されることになっております。

### ミックスされた音源を保存する

```python:room_simulation.py
np.sum(separate_recordings, axis=0)
```

上記の引数内にある、axis=0は行方向に要素が合計されます。
これによって、2つのミックスされた音源が生成されます。

