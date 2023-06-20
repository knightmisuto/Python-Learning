import numpy as np

#θ: theta

#hàm LoadData
def Loadtxt(path):
    try:
        raw = np.loadtxt(path,delimiter = ',')
        X = np.zeros((np.size(raw,0),np.size(raw,1)))
        X[:,0] = 1
        X[:,1:] = raw[:,:-1]
        y = raw[:,-1]
        yield X
        yield y
    except:
        return 0

#Hàm predict: sử dụng để predict output
def predict(X,Theta):
    return X.dot(Theta)

#Hàm Compute cost: để tính J(θ): Hàm J(θ) cho biết độ “phù hợp” của đường thẳng đã tìm so với training set
def computeCost(X,y,Theta):
    predicted = predict(X, Theta)
    sqr_error = (predicted - y)**2
    sum_error = np.sum(sqr_error)
    m = np.size(y)
    j = (1/(2*m))*sum_error
    return j

#Hàm Compute cost Vectorized: để tính J(θ) bằng cách sử dụng vecto
def computeCost_Vec(X,y,Theta):
    error = predict(X, Theta) - y
    m = np.size(y)
    j = (1/(2*m))*np.transpose(error)@error
    return j

#Progress Bar: hiển thị quá trình train
def printProgressBar (iteration, total, suffix = ''):
    percent = ("{0:." + str(1) + "f}").format(100 * ((iteration+1) / float(total)))
    filledLength = int(50 * iteration // total)
    bar = '=' * filledLength + '-' * (50- filledLength)
    print('\rTraining: |%s| %s%%' % (bar, percent), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

#Gradient Descent: Thuật toán được sử dụng để “tìm” ra θ(theta)
#Thuật toán Gradient Descent sẽ chạy nhanh hơn với những vùng dữ liệu nhỏ
def GradientDescent(X,y,alpha=0.02,iter=5000): #giá trị mặc định của alpha = 0.02, số vòng lặp iter tối đa là 5000
    #Giá trị ban đầu của Theta = 0
    theta = np.zeros(np.size(X,1)) #Số lượng theta bằng số cột của X
    #array lưu lại các giá trị j trong quá trình lặp
    J_hist = np.zeros((iter,2)) #kích thước là iter*2, cột đầu chỉ là các số từ 1 đến iter để tiện cho việc plot
                                #kích thước được truyền vào qua một tuple
    #kích thước của training set
    m = np.size(y)
    #ma trận ngược (đảo hàng và cột) của X
    X_T = np.transpose(X)
    #biến tạm để kiểm tra tiến độ Gradient Descent
    pre_cost = computeCost(X,y,theta)
    for i in range(0,iter):
        printProgressBar(i,iter)
        #tính sai số (predict - y)
        error = predict(X,theta) - y
        #thực hiện gradient descent để thay đổi theta
        theta = theta - (alpha/m)*(X_T @ error)
        #Tính J hiện tại
        cost = computeCost(X,y,theta)
        #so sánh với J của vòng lặp trước, so sánh 15 chữ số thập phân
        if round(cost,15) == np.round(pre_cost,15):
            #in ra vòng lặp hiện tại va J
            print('Reach optima at I = %d; J = %.6f'%(i,cost))
            #thêm tất cả các index còn lại sau khi break
            J_hist[i:,0] = range(i,iter)
            #giá trị J sau khi break sẽ như cũ
            J_hist[i:,1] = cost
            #thoát vòng lặp
            break
        #cập nhật pre_cost
        pre_cost = cost
        #lưu lại index vòng lặp hiện tại
        J_hist[i,0] = i
        #lưu lại J hiện tại
        J_hist[i,1] = cost
    yield theta
    yield J_hist

#Hàm Feature Normalize: quy chuẩn các input về một khoảng nhất định để việc training được tối ưu nhất
def Normalize(X):
    #tạo copy của X để không ảnh hưởng trực tiếp đến X
    n = np.copy(X)
    #x0 đầu tiên giá trị = 100
    n[0,0] = 100
    #tính std cho từng feature x
    s = np.std(n,0,dtype= np.float64)
    #tính mean cho tường feature x
    mu = np.mean(n,0)
    n = (n-mu)/s
    #gắn lại x0 = 1
    n[:,0] = 1
    yield n
    yield mu
    yield s

#Normal Equation (phương trình thông thường)
#không cần dùng vòng lặp để tối ưu Theta từng bước (iteration) để tới được Theta tối ưu nhất,
#mà chỉ cần dùng duy nhất 1 biểu thức để tìm trực tiếp Theta, không cần phải thực hiện Feature Normalize.
def NormEqn(X,y):
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)

"""So sánh Normal Equation và Gradient Descent
        Gradient Descent                            Normal Equation
Cần chọn alpha và iter                          Không cần chọn alpha và iter
Cần lặp nhiều bước tối ưu                       Chỉ thực hiện 1 bước
Độ phức tạp O(kn2)                              Độ phức tạp O(n3), cần tìm ma trận khả nghịch
Hoạt động tốt với training set lớn              Xử lí chậm với training set lớn 
                                            (tìm ma trận khả nghịch làm chậm tốc độ khá nhiều)"""