def print_result(n):
    pages = []
    length = 0
    page_count = 0
    # 初始化 pages 页号
    if n % 4 == 0:
        length = n
        for i in range(length + 1):
            pages[i] = i
    else:
        length = ((n >> 2) + 1) << 2
        for i in range(n + 1):
            pages[i] = i
        for i in range(n + 1, length + 1):
            pages[i] = -1
    # 双指针穿插
    print("Printing order for " + n + " pages:")
    left = 1
    right = length
    pl = 0
    pr = 0
