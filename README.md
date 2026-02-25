# Hard ETSP Instance Generator (Graduation Research)

卒業研究で作成した「ユークリッドTSP (ETSP) の“難しい”インスタンスを自動生成する」実験コードです。  
学習は REINFORCE 系（ポリシー勾配）で行い、点群生成ポリシーとして 回転つき異方性GMM（混合ガウス）を用います。  
生成した点群に対して Gurobi（DFJ + Lazy subtour cuts）でETSPを解き、その solver メトリクス（runtime / nodecount / lazy cut回数など）を報酬として最適化します。


## 概要

### 目的
- ETSP に対して、一般的なランダム点群よりも 解くのが難しい点群（インスタンス）を生成する
- “難しさ”は Gurobi の solver から得られるメトリクスで定義し、学習で最大化する

### アプローチ
1. GMMPolicy が `[0,1]^2` 上の点群（n点）をサンプル生成
2. solve_etsp が Gurobi で ETSP を解き、メトリクスを取得
3. メトリクスから 報酬を計算し、REINFORCE でポリシー更新
4. 学習中に点群や学習曲線を保存し、分布の変化を観察する

---

## 2. 主な特徴

- 点群生成ポリシー
  - K成分の混合ガウス（GMM）
  - 各成分が
    - center（中心）
    - scales（異方性スケール）
    - theta（回転）
    - mixture weight（混合比）
    を持つ

- solver
  - Gurobi で ETSPを解く
  - DFJ + Lazy subtour constraints
  - callback で subtour を検出して cut を追加
  - `runtime / nodecount / lazy / mipgap / solved` を記録

- 報酬モード
  - `--reward_mode all` : runtime / nodecount / lazy を混合（+補助特徴を任意で追加）
  - `--reward_mode nodes` : nodecount のみ
  - `--reward_mode lazy` : lazy cut回数のみ
---

## 3. 必要環境

### 必須
- Python 3.10+（推奨）
- numpy
- torch
- matplotlib（画像保存用）

### Gurobi を使う場合
- Gurobi Optimizer
- `gurobipy`
- 有効なライセンス

