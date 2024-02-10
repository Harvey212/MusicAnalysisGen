#please execute in the following format for the folder "inference_code_testing" 
#python mytest.py --weight [model weight path(required)] --valid [valid data folder path(required if you want to generate confusion matrix, t-sne)] --test [test data folder path(required for testing task)] --output [output result folder/filename.csv(required for testing task)]
#Note:
#1.the valid data folder should follow the arrangement: valid data folder/artist folder/album folder/songs of the album
#2.the test data folder should follow the arrangement: test data folder/songs to be tested
#3.the model weight path should be like "./weight folder name/myweight.ckpt" 

#Example:
#If you only want for testing task, you can type "python mytest.py --weight ./myweight/myweight.ckpt --test ./artist20_testing_data/ --output ./save/r11943113.csv"
#If you only want for validation task, you can type "python mytest.py --weight ./myweight/myweight.ckpt --valid ./val/"
#
#