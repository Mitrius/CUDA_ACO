#ifndef ACTUAL_DEVICE_VECTOR
#define ACTUAL_DEVICE_VECTOR
template<class T>
class actual_device_vector {
private:
	T *contents;
	size_t _idx;
	size_t _size;
public:
	actual_device_vector(size_t n){
		contents = new T[n];
		_size=n;
		_idx=0;
	}
	actual_device_vector(){
		contents = new T[10];
		_size=10;
		_idx=0;
	}
	~actual_device_vector(){
		delete[] contents;
	}
	void push_back(T x){
		contents[_idx++]=x;
		if(_idx==_size){
			T *temp = new T[_size*2];
			for(size_t i = 0; i < _size; i++) temp[i]=contents[i];
			delete[] contents;
			contents = temp;
			_size *= 2;
		}
	}
	void erase(size_t n){
		if(n==_idx-1) _idx--;
		else if (_idx==0 || n>_idx) return;
		else contents[n]=contents[--_idx];
		if(_idx<_size/2) {
			T *temp = new T[_size/2];
			for(size_t i = 0; i < _idx; i++) temp[i]=contents[i];
			delete[] contents;
			contents = temp;
			_size /= 2;
		}
	}
	T& operator[](size_t n){
		return contents[n];
	}
	int size(){
		return _idx;
	}
}
#endif