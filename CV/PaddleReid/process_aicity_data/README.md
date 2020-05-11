# The process of aicity data

## Training data
1. convert yml (replace 'gb2312' to 'utf-8') and make aicity_all  and aicity_all/image_train directories
2. generate real_trainval_list.txt and syn_trainval_list.txt using two python scripts
3. based on the path you put images, the user should add the path in each line in real_trainval_list.txt or syn_trainval_list.txt(the reason for this operation is that we use coco pretrained detection model to crop images and these images are in another directory)
4. concatenate two lists
`cat real_trainval_list.txt syn_trainval_list.txt > all_trainval_pids.txt`
5. put aicity_all in dataset directory

## Testing data
1. put image_query and image_test into aicity_all directory
2. `ls image_query/* > query_list.txt `
3. `ls image_test/* > test_list.txt `
4. put list into aicity_all directory