template<class T>
class actual_device_vector {
private:
	T *contents = NULL;
	size_t _idx, _size;
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
		contents[idx++]=x;
		if(i==size){
			T *temp = new T[size*2];
			for(size_t i = 0; i < size; i++) temp[i]=contents[i];
			delete[] contents;
			contents = temp;
			size *= 2;
		}
	}
	void erase(size_t n){
		if(n==i-1) idx--;
		else if (idx==0 || n>idx) return;
		else contents[n]=contents[--idx];
		if(idx<size/2) {
			T *temp = new T[size/2];
			for(size_t i = 0; i < idx; i++) temp[i]=contents[i];
			delete[] contents;
			contents = temp;
			size /= 2;
		}
	}
	T& operator[](size_t n){
		return contents[n];
	}
	int size(){
		return idx;
	}
}