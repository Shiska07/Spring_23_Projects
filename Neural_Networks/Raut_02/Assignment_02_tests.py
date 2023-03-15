import numpy as np
from YourLastName_02_01 import multi_layer_nn_tensorflow

def get_data():
    X = np.array([[0.685938, -0.5756752], [0.944493, -0.02803439], [0.9477775, 0.59988844], [0.20710745, -0.12665261], [-0.08198895, 0.22326154], [-0.77471393, -0.73122877], [-0.18502127, 0.32624513], [-0.03133733, -0.17500992], [0.28585237, -0.01097354], [-0.19126464, 0.06222228], [-0.0303282, -0.16023481], [-0.34069192, -0.8288299], [-0.20600465, 0.09318836], [0.29411194, -0.93214977], [-0.7150941, 0.74259764], [0.13344735, 0.17136675], [0.31582892, 1.0810335], [-0.22873795, 0.98337173], [-0.88140666, 0.05909261], [-0.21215424, -0.05584779]], dtype=np.float32)
    y = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    return (X, y)

def get_data_2():
    X = np.array([[0.55824741, 0.8871946, 0.69239914], [0.25242493, 0.77856301, 0.66000716], [0.4443564, 0.1092453, 0.96508663], [0.66679551, 0.49591846, 0.9536062], [0.07967996, 0.61238854, 0.89165257], [0.36541977, 0.02095794, 0.49595849], [0.56918241, 0.45609922, 0.05487656], [0.38711358, 0.02771098, 0.27910454], [0.16556168, 0.9003711, 0.5345797], [0.70774465, 0.5294432, 0.77920751]], dtype=np.float32)
    y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return (X, y)

def test_random_weight_init():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0, loss='mse')
    assert W[0].dtype == np.float32
    assert W[1].dtype == np.float32
    assert W[0].shape == (3, 8)
    assert W[1].shape == (9, 2)
    assert np.allclose(W[0], np.array([[-0.41675785, -0.05626683, -2.1361961, 1.6402708, -1.7934356, -0.84174734, 0.5028814, -1.2452881], [-1.0579522, -0.9090076, 0.55145407, 2.292208, 0.04153939, -1.1179254, 0.5390583, -0.5961597], [-0.0191305, 1.1750013, -0.7478709, 0.00902525, -0.8781079, -0.15643416, 0.25657046, -0.98877907]], dtype=np.float32))
    assert np.allclose(W[1], np.array([[-0.41675785, -0.05626683], [-2.1361961, 1.6402708], [-1.7934356, -0.84174734], [0.5028814, -1.2452881], [-1.0579522, -0.9090076], [0.55145407, 2.292208], [0.04153939, -1.1179254], [0.5390583, -0.5961597], [-0.0191305, 1.1750013]], dtype=np.float32))

def test_weight_update_mse():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss='mse')
    assert np.allclose(W[0], np.array([[-0.4245687, -0.10408437, -2.1361961, 1.5439054, -1.7934356, -0.84240603, 0.50558764, -1.2452881], [-1.0522263, -0.8971243, 0.55145407, 2.2681074, 0.04153939, -1.1174152, 0.53528196, -0.5961597], [-0.02146104, 1.1607296, -0.7478709, 0.01255831, -0.8781079, -0.15595253, 0.25859874, -0.98877907]], dtype=np.float32))
    assert np.allclose(W[1], np.array([[-0.35916013, -0.00812877], [-2.134117, 1.6413116], [-1.7826873, -0.8358137], [0.5028814, -1.2452881], [-0.94743884, -0.79902333], [0.55145407, 2.292208], [0.04176087, -1.1178356], [0.5722097, -0.5650703], [-0.0191305, 1.1750013]], dtype=np.float32))

def test_weight_update_cross_entropy():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss='cross_entropy')
    assert np.allclose(W[0], np.array([[-0.41575706, -0.0595165, -2.1361961, 1.6398498, -1.7934356, -0.8419575, 0.50609666, -1.2452881], [-1.0587087, -0.90868545, 0.55145407, 2.2924833, 0.04153939, -1.1177626, 0.53695536, -0.5961597], [-0.01939617, 1.1744683, -0.7478709, 0.00885777, -0.8781079, -0.15628049, 0.25784925, -0.98877907]], dtype=np.float32))
    assert np.allclose(W[1], np.array([[-0.41410774, -0.05891695], [-2.1362991, 1.6403737], [-1.7926645, -0.84251845], [0.5028814, -1.2452881], [-1.0575432, -0.9094167], [0.55145407, 2.292208], [0.04151427, -1.1179003], [0.53977644, -0.5968778], [-0.0191305, 1.1750013]], dtype=np.float32))

def test_weight_update_svm():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss='svm')
    assert np.allclose(W[0], np.array([[-0.41470736, -0.06239666, -2.1361961, 1.630932, -1.7934356, -0.842446, 0.5036842, -1.2452881], [-1.0594796, -0.9080366, 0.55145407, 2.2912443, 0.04153939, -1.1173842, 0.53834856, -0.5961597], [-0.01911884, 1.1736286, -0.7478709, 0.00933843, -0.8781079, -0.15592326, 0.25708705, -0.98877907]], dtype=np.float32))
    assert np.allclose(W[1], np.array([[-0.41113284, -0.05189182], [-2.1361961, 1.6407359], [-1.7926713, -0.8408321], [0.5028814, -1.2452881], [-1.0488791, -0.90026903], [0.55145407, 2.292208], [0.04153939, -1.1178389], [0.54191935, -0.59382534], [-0.0191305, 1.1750013]], dtype=np.float32))

def test_assign_weights_by_value():
    (X, y) = get_data()
    W_0 = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]], dtype=np.float32)
    W_1 = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]], dtype=np.float32)
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0, loss='cross_entropy', weights=[W_0, W_1])
    assert np.allclose(W[0], W_0)
    assert np.allclose(W[1], W_1)

def test_error_output_dimensions():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss='mse', validation_split=[0.5, 1.0])
    assert isinstance(err, list)
    assert len(err) == 1
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=3, loss='mse', validation_split=[0.5, 1.0])
    assert isinstance(err, list)
    assert len(err) == 3

def test_error_vals_mse():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4, loss='mse', validation_split=[0.5, 1.0])
    assert np.allclose(err, [8.427941, 6.111313, 4.5837493, 3.5246418])
    (X, y) = get_data_2()
    [W, err2, Out] = multi_layer_nn_tensorflow(X, y, [7, 3], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4, loss='mse', validation_split=[0.5, 1.0])
    assert np.allclose(err2, [3.523861, 2.9595647, 2.5296426, 2.192124])

def test_error_vals_cross_entropy():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4, loss='cross_entropy', validation_split=[0.5, 1.0])
    assert np.allclose(err, [0.72633, 0.7231777, 0.7200506, 0.71694815])
    np.random.seed(5368)
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 4, size=(50, 1))
    y = np.eye(4)[y]
    y = y.reshape(50, 4)
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 4], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4, loss='cross_entropy', validation_split=[0.5, 1.0])
    assert np.allclose(err, [4.146377, 3.9781787, 3.82751, 3.6944547])

def test_initial_validation_output():
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 2], ['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0, loss='cross_entropy', validation_split=[0.5, 1.0])
    assert Out.shape == (10, 2)
    assert np.allclose(Out, np.array([[-1.8369007, -1.7483202], [-1.2605281, -0.8941443], [-1.8605977, -1.5690917], [-2.6287963, -2.4041958], [-3.5842671, -0.94719946], [-2.1864333, -2.2156622], [-4.0781965, -3.561052], [-3.6103907, -2.5557148], [-2.9478502, -0.07346541], [-1.5626245, -1.3875837]], dtype=np.float32))

def test_many_layers():
    np.random.seed(1234)
    (X, y) = get_data()
    [W, err, Out] = multi_layer_nn_tensorflow(X, y, [8, 6, 7, 5, 3, 1, 9, 2], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'], alpha=0.01, batch_size=32, epochs=2, loss='cross_entropy')
    assert W[0].shape == (3, 8)
    assert W[1].shape == (9, 6)
    assert W[2].shape == (7, 7)
    assert W[3].shape == (8, 5)
    assert W[4].shape == (6, 3)
    assert W[5].shape == (4, 1)
    assert W[6].shape == (2, 9)
    assert W[7].shape == (10, 2)
    assert Out.shape == (4, 2)
    assert isinstance(err, list) and len(err) == 2