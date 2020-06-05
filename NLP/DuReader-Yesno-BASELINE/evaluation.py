# coding=utf-8

"""
This module computes evaluation metrics for DuReader-YesNo-Opinion dataset.
"""

import six
import json
import argparse

import zipfile

YESNO_LABELS = {'Yes', 'No', 'Depends'}


def unicode_convert(obj):
    """
    conver the object in to utf-8 encode
    """
    if isinstance(obj, dict):
        return {unicode_convert(key): unicode_convert(value) for key, value in obj.iteritems()}
    elif isinstance(obj, list):
        return [unicode_convert(element) for element in obj]
    elif isinstance(obj, unicode):
        return obj.encode('utf-8')
    else:
        return obj


def data_check(obj):
    """
    check data,For example,check whether the key and the value.

    Raises:
        Raises AssertionError when data is not legal.
    """
    if "id" not in obj:
        raise AssertionError("Missing 'id' field", "101")
    if "yesno_answer" not in obj:
        raise AssertionError("Missing 'yesno_answer' field. id: {}".format(obj['id']), "102")

    if not isinstance(obj["id"], int):
        raise AssertionError(r"'id' must be a int.id: {}".format(obj['id']), "103")
    if not isinstance(obj["yesno_answer"], str):
        raise AssertionError(r"'yesno_answer must be a str.id: {}".format(obj['id']), "104")

    if obj["yesno_answer"] not in YESNO_LABELS:
        raise AssertionError(r"'yesno_answer' must be in set ('Yes', 'No', 'Depends'). \
                id: {}".format(obj['id']), "105")


def read_file(file_name, is_ref=False):
    """
    Read predict answer or reference answer from file

    Args:
        file_name: the name of the file containing predict result or reference
                    result.

    Raises:
        Raises ValueError when data format is not right

    Returns:
        A dictionary mapping question_id to the result information.
    """

    def _open(file_name, mode, zip_obj=None):
        if zip_obj is not None:
            return zip_obj.open(file_name, mode)
        return open(file_name, mode)

    results = dict()

    zf = zipfile.ZipFile(file_name, 'r') if file_name.endswith('.zip') else None
    file_list = [file_name] if zf is None else zf.namelist()

    for fn in file_list:
        for line in _open(fn, "r", zip_obj=zf):
            try:
                obj = json.loads(line.strip())
                if six.PY2:
                    obj = unicode_convert(obj)
            except ValueError:
                raise ValueError("Every line of data should be legal json", "106")

            data_check(obj)

            qid = obj["id"]
            yesno_answer = obj["yesno_answer"]

            if qid in results:
                raise AssertionError("Duplicate id: {}".format(qid), "107")

            results[str(qid)] = yesno_answer

    return results


def compute_acc(ref_dict, pred_dict):
    """
    Compute acc score

    Args:
        predict_dict:the predict dictonary mapping the id to the predict result.
        ref_dict:the test dictonary mapping the id to the orgin result.
    """
    if set(pred_dict.keys()) != set(ref_dict.keys()):
        raise AssertionError("missing keys, id: {}".format(
            list((set(ref_dict.keys()) ^ set(pred_dict.keys())))[0]), "108")

    score = {}
    right = 0
    for key, value in ref_dict.items():
        if pred_dict[key] == value:
            right += 1

    score["acc"] = right * 1.0 / len(ref_dict)

    return score


def format_metrics(metrics, err_msg):
    """
    Format metrics

    Args:
        metrics: A dict object contains metrics for different tasks.
        err_msg: Exception raised during evaluation.

    Returns:
        Formatted result.
    """
    result = {}
    if err_msg is not None:
        if len(list(err_msg)) >= 2:
            return {'errorMsg': str(err_msg[0]), 'errorCode': str(err_msg[1]), 'data': []}
        else:
            print(err_msg)
            return {'errorMsg': str(err_msg), 'errorCode': "109", 'data': []}

    data = []
    for name, value in metrics.items():
        obj = {
            "name": name,
            "value": round(value * 100, 2),
        }
        data.append(obj)

    result["data"] = data
    result["errorCode"] = "0"
    result["errorMsg"] = "success"
    return result


def main(args):
    """
    Do evaluation.
    """

    err = None
    metrics = {}
    try:
        pred_result = read_file(args.ref_file)
        with open(args.pred_file) as fin:
            ref_result = json.load(fin)

        metrics = compute_acc(ref_result, pred_result)

    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae

    print(json.dumps(
        format_metrics(metrics, err),
        ensure_ascii=False).encode("utf-8"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ref_file', help='reference file')
    parser.add_argument('pred_file', help='predict file')

    args = parser.parse_args()
    main(args)
