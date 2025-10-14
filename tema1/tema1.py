import pathlib

# 1
def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace(" ", "").strip() #curata linia de spatii, inclusiv de cele de la inceput si sfarsit
            left, right = line.split("=")
            B.append(float(right)) 
            
            coef = []
            for var in ["x", "y", "z"]:
                if var in left:
                    part = left.split(var)[0]
                    if '+' in part:
                        sign = 1
                        part = part.split('+')[-1]
                    elif '-' in part:
                        sign = -1
                        part = part.split('-')[-1]
                    else:
                        sign = 1

                    if part == "":
                        num = 1.0
                    else:
                        num = float(part)
                    
                    coef.append(sign * num)

                    left = left.split(var, 1)[1]
                else:
                    # variabila nu apare, deci coeficientul ei e 0
                    coef.append(0.0)
            A.append(coef)

    return A, B

A, B = load_system(pathlib.Path("system.txt"))
print("A =", A)
print("B =", B)


# 2.1
def determinant(matrix: list[list[float]]) -> float:
    a11 = matrix[0][0]
    a12 = matrix[0][1]
    a13 = matrix[0][2]    
    a21 = matrix[1][0]
    a22 = matrix[1][1]
    a23 = matrix[1][2]    
    a31 = matrix[2][0]
    a32 = matrix[2][1]
    a33 = matrix[2][2]

    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )

    return det

print(f"{determinant(A)=}")


# 2.2
def trace(matrix: list[list[float]]) -> float:
    a11 = matrix[0][0]
    a22 = matrix[1][1]
    a33 = matrix[2][2]

    trace = a11 + a22 + a33

    return trace

print(f"{trace(A)=}")


# 2.3
def norm(vector: list[float]) -> float:
    norma = vector[0]**2 + vector[1]**2 + vector[2]**2
    norma = norma**(0.5)

    return norma

print(f"{norm(B)=}")


# 2.4
def transpose(matrix: list[list[float]]) -> list[list[float]]:
    tmatrix = [row[:] for row in matrix]

    tmatrix[0][1], tmatrix[1][0] = tmatrix[1][0], tmatrix[0][1]
    tmatrix[0][2], tmatrix[2][0] = tmatrix[2][0], tmatrix[0][2]
    tmatrix[1][2], tmatrix[2][1] = tmatrix[2][1], tmatrix[1][2]

    return tmatrix

print(f"{transpose(A)=}")


# 2.5
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    mmatrix = []

    for i in range(0,3):
        s = 0
        for j in range (0,3):
            s+= matrix[i][j] * vector[j]
        mmatrix.append(s)

    return mmatrix

print(f"{multiply(A, B)=}")


# 3
def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det = determinant(matrix)

    if(det == 0):
        print("Sistemul nu are solutie unica.")
        return
    
    mx = [[vector[0], matrix[0][1], matrix[0][2]],
          [vector[1], matrix[1][1], matrix[1][2]],
          [vector[2], matrix[2][1], matrix[2][2]]
        ]   
    
    my = [[matrix[0][0], vector[0], matrix[0][2]],
          [matrix[1][0], vector[1], matrix[1][2]],
          [matrix[2][0], vector[2], matrix[2][2]]
        ]   
        
    mz = [[matrix[0][0], matrix[0][1], vector[0]],
          [matrix[1][0], matrix[1][1], vector[1]],
          [matrix[2][0], matrix[2][1], vector[2]]
        ]   

    x = determinant(mx)/det
    y = determinant(my)/det
    z = determinant(mz)/det

    #return (x, y, z)
    return [round(x,4), round(y,4), round(z,4)]

print(f"{solve_cramer(A, B)=}")


# 4 
def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    # elimin linia i si coloana j
    return [[matrix[a][b] for b in range(0,3) if b != j] for a in range(0,3) if a != i]  

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    Cof = [[0.0 for i in range(0,3)] for j in range(0,3)]
    for i in range(0,3):
        for j in range(0,3):
            m = minor(matrix, i, j)
            Cof[i][j] = ((-1) ** (i + j)) * (m[0][0] * m[1][1] - m[0][1] * m[1][0])
    return Cof

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    Cof = cofactor(matrix)
    return transpose(Cof)

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det = determinant(matrix) 

    if(det == 0):
        print("Sistemul nu are solutie unica si matricea A nu este inversabila.")
        return
    
    a = 1/det
    adj = adjoint(matrix)
    Inv = [[adj[i][j] * a for j in range(0,3)] for i in range(0,3)]
    return multiply(Inv, vector)

print(f"{solve(A, B)=}")