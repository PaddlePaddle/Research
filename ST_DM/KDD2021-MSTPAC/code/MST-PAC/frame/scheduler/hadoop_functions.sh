#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# change the file directory to a uniform format
# for example ////home/////work////xx/////./../ ==> /home/work
# 1. delete continuous /
# 2. delete \r or \n or / in the end
# 3. delete ./ and /./
# 4. delete /../
function standard_dir()
{
    local dirname=$1
    local orig=`echo ${dirname} | \
                  sed 's/\/\+/\//g' | \
                  sed 's/\r$//g' | \
                  sed 's/\(^[^/]\)/\.\/\1/g'`
    local new=""
    local orig1=""
    while [[ 1 ]] ;do
        orig1=`echo ${orig} | sed 's/\/\.\//\//g'`
        new=`echo ${orig1} | sed 's/\/[^/.]\+\/\.\.//g'`
        if [ X"${new}" = X"${orig}" -o X"${new}" = X ];then
            new=`echo ${new} | \
                  sed 's/^\.\///g' | \
                  sed 's/\/\.$//g' | \
                  sed 's/\/$//g'`
            break;
        else
            orig="${new}"
        fi
    done

    if [ X${new} = X ];then
        new='.'
    fi
    echo ${new}
}

# put local file to hdfs
# $1 hadoop_home
# $2 fs
# $3 ugi
# $4 localfile, file may be directory
# $5 remotedir, localfile will put to remotedir/localfile in hdfs
# $6 dfs.use.native.api=0 need afs-agent, and need dfs.agent.port
# $7 dfs_agent_port, the port of afs-agent
function hadoop_put_file()
{
    local hadoop_home=$1
    local fs=$2
    local ugi=$3
    local localfile=`standard_dir $4`
    local remotedir=`standard_dir $5`
    local dfs_use_native_api=$6
    local dfs_agent_port=$7
    local max_retry=5
    local retry=0
    local ret=1
    local message=""
    while ((retry < max_retry));do
        if [ ! -f ${localfile} -a ! -d ${localfile} ]; then
            #log_fatal
            echo "$localfile not a file or directory" >&2
        fi
        if [[ ${dfs_use_native_api} == "" ]] || [[ ${dfs_agent_port} == "" ]]; then
            ${hadoop_home}/bin/hadoop dfs -Dfs.default.name=${fs} \
                                      -Dhadoop.job.ugi=${ugi} \
                                      -put ${localfile} ${remotedir}

        else
            ${hadoop_home}/bin/hadoop dfs -Dfs.default.name=${fs} \
                                      -Dhadoop.job.ugi=${ugi} \
                                      -Ddfs.use.native.api=${dfs_use_native_api} \
                                      -Ddfs.agent.port=${dfs_agent_port} \
                                      -put ${localfile} ${remotedir}
        fi
        ret=$?
        if [ ${ret} -ne 0 ];then
            message="Warning: upload to [${remotedir}] failed"
            message="$message with code[${ret}], retry[${retry}]"
            #log_info 
            echo "$message" >&2
            ((retry++))
            ${hadoop_home}/bin/hadoop dfs -Dfs.default.name=${fs} \
                                          -Dhadoop.job.ugi=${ugi} \
                                          -rm ${remotedir}/`basename ${localfile}`
            continue
        else
            #log_info 
            echo "upload ${localfile} to [${remotedir}] success" >&2
            break
        fi
    done
    return ${ret}
}

# get remotefile from hdfs to local
# $1 hadoop_home
# $2 fs
# $3 ugi
# $4 remotefile this may be a directory
# $5 localdir
# $6 dfs.use.native.api=0 need afs-agent, and need dfs.agent.port
# $7 dfs_agent_port, the port of afs-agent
function hadoop_get_file()
{
    echo "start hadoop_get_file $4" >&2
    local hadoop_home=$1
    local fs=$2
    local ugi=$3
    local remotefile=`standard_dir $4`
    local localdir=`standard_dir $5`
    local dfs_use_native_api=$6
    local dfs_agent_port=$7
    #echo "localdir=$localdir"
    local max_retry=5
    local retry=0
    local ret=1
    local message=""
    ${hadoop_home}/bin/hadoop dfs -Dfs.default.name=${fs} \
                                  -Dhadoop.job.ugi=${ugi} \
                                  -Ddfs.use.native.api=${dfs_use_native_api} \
                                  -Ddfs.agent.port=${dfs_agent_port} \
                                  -test -e ${remotefile}
    if [ $? -ne 0 ];then
        #log_fatal 
        echo "hadoop dfs -test -e [${remotefile}] failed" >&2
    fi
    while ((retry < max_retry));do
       ${hadoop_home}/bin/hadoop dfs -Dfs.default.name=${fs} \
                                     -Dhadoop.job.ugi=${ugi} \
                                     -Ddfs.use.native.api=${dfs_use_native_api} \
                                  	 -Ddfs.agent.port=${dfs_agent_port} \
                                     -get ${remotefile} ${localdir}
       ret=$?
       if [ ${ret} -ne 0 ];then
           message="Warning: download from [${remotefile}]"
           message="$message failed with code[${ret}], retry[${retry}]."
           #log_info 
           echo "$message" >&2
           ((retry++))
           rm -rf ${localdir}/`basename ${remotefile}`
           continue
       else
           #log_info 
           echo "download from [${remotefile}] to ${localdir} success" >&2
           break
       fi
    done
    return ${ret}
}

# generate file list
# $1 hadoop_home
# $2 fs_name
# $3 fs_ugi
# $4 remote file(this may be a directory)
# $5 file list
# $6 use_hadoop_vfs: True or False
# $7 dfs.use.native.api=0 need afs-agent, and need dfs.agent.port
# $8 dfs_agent_port, the port of afs-agent
function hadoop_ls_file()
{
    local hadoop_home="$1"
    #echo "current hadoop_home=========$hadoop_home" >&2
    local fs_name="$2"
    local fs_ugi="$3"
    local remote_file="$4"
    local file_list="$5"
    local use_hadoop_vfs="$6"
    local dfs_use_native_api=$7
    local dfs_agent_port=$8
    if [ "${remote_file}" = "None" ]; then
        cat /dev/null > ${file_list}
    else
        if [ "${use_hadoop_vfs}" = "True" ]; then
            # Access data via hadoop vfs
            echo ${remote_file} | awk 'BEGIN{RS=" ";} {print}' \
                | xargs -i ls -l --time-style=long-iso {} > ${file_list}
            sed -i '/^ *$/d' ${file_list}
        else
            ${hadoop_home}/bin/hadoop dfs -Dfs.default.name=${fs_name} \
                                          -Dhadoop.job.ugi=${fs_ugi} \
                                          -Ddfs.use.native.api=${dfs_use_native_api} \
                                          -Ddfs.agent.port=${dfs_agent_port} \
                                          -ls ${remote_file} \
                | egrep -v '^Found.*items$' > ${file_list}
        fi
    fi
    return $?
}

# check path valid
# $1 hadoop_home
# $2 fs_name
# $3 fs_ugi
# $4 remote file(this may be a directory)
# $5 dfs.use.native.api=0 need afs-agent, and need dfs.agent.port
# $6 dfs_agent_port, the port of afs-agent
function hadoop_check_path() {
    local hadoop_home="$1"
    local fs_name="$2"
    local fs_ugi="$3"
    local remote_file="$4"
    local dfs_use_native_api=$5
    local dfs_agent_port=$6
    ${hadoop_home}/bin/hadoop dfs -Dfs.default.name=${fs_name} \
            -Dhadoop.job.ugi=${fs_ugi} \
            -Ddfs.use.native.api=${dfs_use_native_api} \
            -Ddfs.agent.port=${dfs_agent_port} \
            -ls ${remote_file}
    return $?
}
