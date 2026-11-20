# to-do list
- find the common usage of *accuracy retention*
- find the common name of *training setting*
- use a better name of soy+cotton and other dataset
- use another font to the specific noun we use like acc retention
- reduce the infor in the table, just show the important info 
# intro
## From the fgir dataset to quantization
fgir is a difficult problem in image recognition

-> need larger model to have better performance (need to find the paper to cite)

-> due to the highly ram cost and the computation resource on edge device, we need to do quantization 

-> two way to do quantization PTQ and QAT, althought QAT usually have better performance, the training process is need make it bacome harder to implement when we lack of computing resource or dataset 

-> for the PTQ, we only need some calibration data or even not use calibration data, just like the HQQ we use in this paper. It provide a fast way to reduce the model size and still have acceptable performance

->There are many research focus on how to improve the PTQ acc, and how will PTQ be affected by the backbone model and other aspect like the class difference in one dataset.In these research, we can find in same model and quantization setting, the acc retentation still differ from differnet dattset. This phenomenon lead us to the **first quaestion : will the dataset affect the quantization robustness?** 

-> to analysis that problem, we do a series or experiment including 16 dataset / 3 backbone / 3 training setting / 8 different weight bits / 4 different group size

-> as the fig 1(the box graph, y axis for acc retention, two box for soy+cotton / other) show, we can see the appearent different from these two different category of dataset, this make us to dive depper and think the **second problem : how and why dataset affect quantizatio robustness**

-> to solve this problem we use different way to solve the problem : (list the analysis we made). Overall, our contribution answer the following problem

- We found out that the dataset have appearent connection with quantization. In same backbone and training setting, the ranking of dataset have a robustness ranking. For the different backbone and setting, the ranking still shows a strong robustness (I know that it might only have phi=0.6 for this part, but I think this analysis way have some problem like the other dataset might change ranking but not having a vary appearent differnce in these dataset, cann you give me some advice)
- We analysis the result from different aspect and different way, finding out the most appearent connection is logits distribution and margin. When we make the same distribution to other dataset and soy+cotton dataset, the gap have a appearent closure, which means the logits have a large connection with the dataset and acc retention

# experiment setup

## dataset
## model
## quantiztion setting
## metric 

# result
## will the dataset affect quantiation
## how and why will dataset affect quantization
### logits analysis

# conclusion

# appendix
