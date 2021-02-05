
function get_script_dir()
{
    local script=$1
    local script_abs_path=`readlink -f $script`
    local script_abs_dir=`dirname $script_abs_path`
    echo "$script_abs_dir"
}

cur_script_dir=`get_script_dir $0`

python $cur_script_dir/reducer.py | python $cur_script_dir/reducer2.py
