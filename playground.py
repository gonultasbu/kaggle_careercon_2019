import numpy as np 

def balanced_cv(conf_mat):
    assert len(conf_mat.shape) == 2
    rows, columns = conf_mat.shape
    intra_class_scores = np.zeros((1, columns))
    column_sums = np.sum(conf_mat, axis=0)
    for c in range(columns):
        intra_class_scores[0][c] = float(conf_mat[c][c])/column_sums[c]
        
    balanced_score = np.mean(intra_class_scores)
    return balanced_score

if __name__ == "__main__":
    cm = np.array([[3,4,5],[6,7,9],[4,8,4]])
    balanced_score, class_weights = balanced_cv(cm)
    print(balanced_score)
    print(class_weights)

