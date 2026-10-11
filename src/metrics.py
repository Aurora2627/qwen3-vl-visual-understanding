"""PyTorch metrics retain abstentions and malformed answers in the denominator."""
import torch


def summarize(rows):
    if not rows:
        raise ValueError('Empty evaluation')
    labels = ['PASS', 'FAIL', 'UNCERTAIN', 'INVALID']
    matrix = torch.zeros((2, 4), dtype=torch.int64)
    for row in rows:
        if type(row['expected']) is not bool:
            raise ValueError('Expected a boolean reference')
        matrix[0 if row['expected'] else 1, labels.index(row['verdict'])] += 1
    n = int(matrix.sum())
    tp, fn = int(matrix[0, 0]), int(matrix[0, 1])
    fp, tn = int(matrix[1, 0]), int(matrix[1, 1])
    answered = tp + tn + fp + fn
    groups = {}
    for row in rows:
        groups.setdefault(row['image_group'], []).append(row)
    pairs = [pair for pair in groups.values() if len(pair) == 2 and {x['expected'] for x in pair} == {True, False}]
    pair_correct = sum(all(x['verdict'] == ('PASS' if x['expected'] else 'FAIL') for x in pair) for pair in pairs)
    result = {'n': n, 'accuracy_all': (tp+tn)/n, 'coverage': answered/n,
              'accuracy_answered': (tp+tn)/answered if answered else None,
              'invalid_rate': int(matrix[:, 3].sum())/n,
              'uncertain_rate': int(matrix[:, 2].sum())/n,
              'false_accept_count': fp, 'false_reject_count': fn,
              'paired_images': len(pairs), 'both_pair_members_correct': pair_correct,
              'pair_accuracy': pair_correct/len(pairs) if pairs else None,
              'always_pass_accuracy': sum(x['expected'] for x in rows)/n,
              'always_fail_accuracy': sum(not x['expected'] for x in rows)/n,
              'confusion_rows': ['reference_PASS', 'reference_FAIL'],
              'confusion_columns': labels, 'confusion': matrix.tolist()}
    return result, matrix
