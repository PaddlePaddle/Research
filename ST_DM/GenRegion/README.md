# Introduction
Urban area of interest (AOI) is broadly defined as the area within an urban environment that attracts People’s attention, and it is the prerequisite for many spatio-temporal applications such as urban planning and traffic analysis. However, constructing closed polygons to form numerous AOIs in a city is time-consuming and perplexing. Traditional grid-based methods offer the simple solution as they cut the city into equal-sized grids, but information about the urban structure will be lost. To relieve this problem, we report a vector-based approach to segment urban space into proper regions using road networks. The algorithm first tries to simplify the road network by conducting hierarchical clustering on two endpoints of each road segment. Then, every segment is broken into pieces by its intersections with other segments to guarantee that each segment only connects to others at the start and end nodes. After that, we generate each region by recursively finding the leftmost link of a segment until a link has travelled back to itself. Lastly, we merge tiny blocks and remove sub-regions to have a more meaningful output. To ensure the robustness of our model, we use road data from various sources and generate regions for different cities, and we also compare our model to other urban segmentation techniques to demonstrate the superiority of our method. A coherent application public interface is public available, and the code is open sourced.<br> <br>
Raw road network and generated regions of New York  
<p align="center">
    <img align="center" src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/newyork_link.png" width="400" height="400" alt="newyork_link" style="margin:0 auto"/>
    <img align="center" src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/new york colored.png" width="400" height="400" alt="newyork_polygons" style="margin:0 auto"/>  
</p>
<br> Raw road network and generated regions of Beijing  
<p align="center">
<img align="center" src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/beijing_road_network.png" width="400" height="400" alt="processing" style="text-align:center"/>
<img align="center" src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/beijing colored.png" width="400" height="400" alt="processing"/>  
</p>
<br> Raw road network and generated regions of Shanghai  
<p align="center">
<img align="center" src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/shanghai_links.png" width="400" height="400" alt="processing" style="text-align:center"/>
<img align="center" src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/shanghai colored.png" width="400" height="400" alt="processing"/>  
</p>
<br>Example of zhongguancun softwore park(zpark).  
<p align="center">
<img align="center" src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/ZPark.png" width="500" height="400" alt="Result of ZPARK" style="margin:0 auto"/>  
</p>  

## Main advantages:  

* Efficient (short run time).   
* Full division (only polygons in the output).
* Flexible interface.  

## Download the urban road network dataset
https://figshare.com/articles/dataset/Urban_Road_Network_Data/2061897  
Dataset posted on 20.01.2016, 02:21 by Urban Road Networks.   
The website offers road network datasets for 80 most populated urban areas in the world. The data for each city consists of a graph edge list and two corresponding GIS shapefiles (i.e., links and nodes).

## Get Start
1. Clone the code from github <br>. 
2. Get the urban road network from the website, one possible is https://figshare.com/articles/dataset/Urban_Road_Network_Data/2061897.  
3. Run the program.
```
cd ./Research/ST_DM/GenRegion/src  
python run.sh --in-file --out-file   
```
>    --in-file the file path of dataset of urban road network <br> 
>    --out-file the result path of blocks that the program deal segment urban space into proper regions using road networks <br>
>   the log message:
  <img src="https://github.com/PaddlePaddle/Research/blob/master/ST_DM/GenRegion/result/process.png" width="400" height="200" alt="processing"/>  
4. When the "yyyy-mm-dd hh:mm:ss finished " log message appears, the entire program finishes. The result file is in --out-file path.
