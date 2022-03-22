import os
import time
from multiprocessing import Process, Queue


def counter(number, que):
    print("Process ID : ", os.getpid())  # process id 출력
    sum = 0
    start = number[0]
    end = number[1]

    ## start 부터 end 까지의 합을 구한다
    for i in range(start, end+1):
        sum += i

    que.put(sum)


def main():
    start_time = time.time()  # 시작 시간 측정 시작
    que = Queue()
    process_cnt = 2  # 몇개의 프로세스로 동작시킬지
    instruct = []  # 각각의 프로세스에 넣을 parameter array
    length = int(100000/process_cnt)

    for i in range(process_cnt):
        instruct.append([i*length+1, (i+1)*length])

    procs = []  # process 들을 저장할 array

    for index, number in enumerate(instruct):
        proc = Process(target=counter, args=(number, que))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    print("%s seconds" % (time.time() - start_time))  # 걸린 시간 출력
    total = 0
    que.put('exit')
    while True:
        tmp = que.get()
        if tmp == 'exit':
            break
        else:
            total += tmp
    print("Result : ", total)

if __name__ == '__main__':
    main()
