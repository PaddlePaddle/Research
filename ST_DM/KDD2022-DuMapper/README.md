
## DuMapper: Towards Automatic Verification of Large-Scale Points of Interest with Street Views at Baidu Maps

### Abstract
With the increased popularity of mobile devices, web mapping services have become an indispensable tool in our daily lives. To provide user-satisfied services like location searches, the point of interest (POI) database is the fundamental infrastructure, as it archives multimodal information on billions of geographic locations closely related to people's lives, such as a shop or a bank. Therefore, it plays a vital role in verifying the correctness of a large-scale POI database. To achieve this goal, many industrial companies adopt volunteered geographic information (VGI) platforms that enable thousands of crowdworkers and expert mappers to verify POIs seamlessly but have to spend millions of dollars every year. 
To save the tremendous labor costs, we devised DuMapper, an automatic system for large-scale POI verification with the multimodal street-view data at Baidu Maps. This paper presents not only DuMapper I, which imitates the process of POI verification conducted by expert mappers but also proposes DuMapper II, a highly efficient framework to accelerate POI verification by means of deep multimodal embedding and approximate nearest neighbor (ANN) search. DuMapper II takes the signboard image and the coordinates of a real-world place as input to generate a low-dimensional vector, which can be leveraged by ANN algorithms to conduct a more accurate search through billions of archived POIs in the database for verification within milliseconds. Compared with DuMapper I, experimental results demonstrate that DuMapper II can significantly increase the throughput of POI verification by 50 times. DuMapper has already been deployed to production since June 2018, which dramatically improves the productivity and efficiency of POI verification at Baidu Maps. As of December 31, 2021, it has made over 405 million times of POI verification within a 3.5-year period, representing an approximate workload of 800 high-performance expert mappers. We have also released the source code of DuMapper II at Github to both research and industrial communities for reproducibility tests.

### How to run
train.py: 

infer.py:

DME:

ANN:
