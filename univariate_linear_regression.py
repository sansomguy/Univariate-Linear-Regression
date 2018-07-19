from numpy import *

def loss_function(m_gradient, b_gradient, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m_gradient*x + b_gradient))**2
    return totalError/ float(len(points))

def step_gradient(m_current, b_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
         X = points[i,0]
         Y = points[i,1]
         m_gradient += -(2/N)*X*(Y - (m_current*X + b_current))
         b_gradient += -(2/N)*(Y - (m_current*X + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_m, new_b]

def gradient_descent(m_initial, b_initial, points, learningRate, iterations):
    m = m_initial
    b = b_initial
    for i in range(iterations):
        m, b = step_gradient(m,b, array(points), learningRate)
    return [m, b]
        
def run():
    #Train our model on some data.
    points = genfromtxt("data.csv", delimiter=",")
    initial_m = 0
    initial_b = 0
    learningRate = 0.0001
    learningTime = 1000
    
    initial_loss = loss_function(initial_m, initial_m, points)
    print("Initial: loss {0}, b =  {1}, m = {2}".format(initial_loss, initial_b, initial_m))
    
    # Learn through gradient descent
    [m, b] = gradient_descent(initial_m, initial_b, points, learningRate, learningTime)

    after_learning_loss = loss_function(m, b, points)
    print("After learning: loss {0}, b = {1}, m = {2}".format(after_learning_loss, b, m))



if __name__ == '__main__':
    run()