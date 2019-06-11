# positioning_RNN
3D VLP using regression neural network

1. data_WKNN
    Locate the camera using WKNN(weighted K-nearest neighbor) in same dataset. (It's useless because of the very lager positioning error)
    
    WKNN.py-main function using WKNN method
    W_loss.txt--the MSE (mean square error) with different K's value.
    MPE.csv--the mean positioning error with different K. (The difference of MPE and MSE is showed in my graduate thesis.)
    height_error.py--plot the picture of "height--MPE".
    other *.csv files--test and training dataset.
    
2. keras_main
    Locate the camera using regression neural network in same dataset.(the input is the coordinate of the LED's image point, and the output is the position of the camera)
    
    keras_RMSprop_final.py--A full-connected network is used to estimate the camera's coordinate with RMSprop optimizer.
    keras_adam_final.py--A full-connected network is used to estimate the camera's coordinate with adam optimizer.
    keras_load_evaluate.py--Can estimate the output of test dataset with saved model， and calculate the MPE.
    height_error.py--plot the picture of "height--MPE".
    *.csv--test and training dataset.
    
    BestModel--Save the best network model with adam and RMSprop optimizer.
    plot_loss_value--Save the loss in training process.
    
3. matlab_tools
    Locate the camera using geometrical method to campare with RNN method. (Which is showed in my graduate thesis.)
    
    camera_3D_1.m--Generate the dataset for positioning, Which the above "same dataset" refers to.
    get_image_point_coordinate--Input the LED's position and 
    fun_position.m--Obtain the cooridinate of camera by newton method.
    fun_forK.m--Obtain the Jacobian matrix in newton method.
    plotwcs.m--plot the LED, camera, image point in WCS
    *.csv--The dataset generated by camera_3D_1.m
    
