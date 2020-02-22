import numpy as np

def print_matrix(state):
    counter = 0
    for row in range(0, len(state), 3):
        if counter == 0 :
            print("-------------")
        for element in range(counter, len(state), 3):
            if element <= counter:
                print("|", end=" ")
            print(int(state[element]), "|", end=" ")
        counter = counter +1
        print("\n-------------")

def check_solvable(input_state):
    a = input_state.flatten()
    index = np.where(a==0)
    a = np.delete(a,index)
    insolve = 0
    idx = 0

    for i in a:
        idx = idx + 1
        for j in range(idx, len(a)):
            if i>a[j]:
                insolve = insolve + 1

    if insolve%2 == 0:
        return True

    else:
        return False

def move(state,action):
        pass

def BlankTileLocation(state):
    a = state.flatten()
    index = np.where(a==0)
    index = index[0]
    j = int((index)%3 + 1)
    i = int(index/3) + 1

    return [i, j]  

def swap(state, old_i, old_j, new_i, new_j):

    old = state[old_i-1, old_j-1]

    state[old_i-1, old_j-1] = state[new_i-1, new_j-1]

    state[new_i-1, new_j-1] = old   

    return state

def ActionMoveLeft(state):

    [i, j] = BlankTileLocation(state)

    status  = False
    new_state = np.copy(state)
    
    if j>1:
        new_state = swap(new_state, i, j, i, j-1)
        status = True

    return [status, new_state]

def ActionMoveRight(state):

    [i, j] = BlankTileLocation(state)

    status  = False

    new_state = np.copy(state)

    if j<3:
        new_state = swap(new_state, i, j, i, j+1)
        status = True

    return [status, new_state]

def ActionMoveUp(state):

    [i, j] = BlankTileLocation(state)

    new_state = np.copy(state)

    status  = False
    
    if i>1:
        new_state = swap(new_state, i, j, i-1, j)
        status = True

    return [status, new_state]

def ActionMoveDown(state):

    [i, j] = BlankTileLocation(state)

    new_state = np.copy(state)

    status  = False
    
    if i<3:
        new_state = swap(new_state, i, j, i+1, j)
        status = True

    return [status, new_state]

def statecheck(state, states):

    for key in states.keys():
        if np.array_equal(state, states[key][0]):
            return False
    else:
        return True

def addstate(state, states, prev_state):

    global statenum, inter

    # Storing the state as a string key to optimize for time!
    state_str = str(state.flatten())

    states[state_str] = [state, prev_state]

    fname1 = 'NodesInfo.txt'

    with open(fname1, "a+") as myfile:
        print(str(inter) + " " + str(statenum))
        myfile.write(str(inter) + " " + str(statenum) + "\n")

    statenum = statenum + 1

    return states


def compute_states(init_state):

    global statenum, states, goal_state, action

    solvable = check_solvable(init_state)

    if solvable == False:
        print("The cube is not solvable from this state.")
        return states

    elif solvable == True:
        pass
        # print("The cube is solvable!")
    loopnum = 0

    state = init_state

    status_all = True
    checker1 = False
    checker2 = False
    checker3 = False
    checker4 = False

    while status_all!= False:
        loopnum = loopnum + 1

        [status1, state_cur] = ActionMoveUp(state)

        if status1 != False:
            checker1 = statecheck (state_cur, states)
            if checker1 == True:
                states = addstate(state_cur, states, state)
                action = int(str(action) + str(1))

        [status2, state_cur] = ActionMoveLeft(state)

        if status2 != False:
            checker2 = statecheck (state_cur, states)
            if checker2 == True:
                states = addstate(state_cur, states, state)
                action = int(str(action) + str(2))

        [status3, state_cur] = ActionMoveRight(state)

        if status3 != False:
            checker3 = statecheck (state_cur, states)
            if checker3 == True:
                states = addstate(state_cur, states, state)
                action = int(str(action) + str(3))

        [status4, state_cur] = ActionMoveDown(state)

        if status4 != False:
            checker4 = statecheck (state_cur, states)
            if checker4 == True:
                states = addstate(state_cur, states, state)
                action = int(str(action) + str(4))

        status_all = checker1 or checker2 or checker3 or checker4

    return states

def generate_path(states, inter):
    key = str(goal_state.flatten())
    fname = 'nodePath.txt'
    parentarr = []

    print("Traceback is: ")
    for i in range(0,inter):
        parent = states[key][1]
        print(parent)
        key = str(parent.flatten())
        parentarr.append(parent.transpose().flatten())

    np.savetxt(fname,parentarr, fmt = "%d")

    return 0

def main():
    global states, stateaction, inter

    states[str(init_state.flatten())] = [init_state]
    found_goal = 0

    while (found_goal != 1):
        inter = inter + 1
        for key in states.copy().keys():
            states = compute_states(states[key][0])

        for key in states.copy().keys():
            if np.array_equal(goal_state, states[key][0]):
                print("Found goal state in {} steps!".format(inter))
                found_goal = 1

                generate_path(states, inter)

        if inter>15:
            break

    statearr = []
    fname = 'Nodes.txt'
    for key in states.copy().keys():
        statearr.append(states[key][0].transpose().flatten())
    
    np.savetxt(fname, statearr, fmt='%d')

    return 0

if __name__ == '__main__':
    inter = 0
    action = 0

    # THE INIT STATE NEEDS TO BE CHANGED HERE
    init_state = np.array([[1, 2, 3], [4, 5, 6], [0, 7, 8]])

    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

    states = {}
    statenum = 1
    main()
