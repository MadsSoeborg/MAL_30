def transpose(matrix: list[list[int]]) -> list[list[int]]: 
    if not matrix:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] 
            for i in range(len(matrix[0]))]

def is_valid_matrix(matrix: list[list[int]]) -> bool:
    if not matrix or not isinstance(matrix, list):
        return False
    row_length = len(matrix[0])
    return all(isinstance(row, list) and len(row) == row_length 
              for row in matrix)