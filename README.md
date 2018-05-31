```python
train_features = train_data["features"]
```
##### 여기서 이런 에러가 계속 뜬다.
#####  IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

##### 에러가 뜨는 이유는 train_data의 features를 key가 아닌 index로 인식한다는 것이다. 즉, npy에 저장할 때는 dictionary로 저장했지만 불러올때는 npy array로 불러오게 된다. 다음과 같이 해결하면 된다.
```python
train_data = np.load("blood_cell_train.npy") 
train_data = train_data.item()
```