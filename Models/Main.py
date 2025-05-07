import importlib

modules = {
    1: 'LogisticR',
    2: 'multiLR',
    3: 'SimpleLR',
    4: 'PolyRv2'
}

while True :

    try:
        wh = int(input("Choose an option:\n1: Logistic Regression\n2: Multiple Linear Regression\n3: Simple Linear Regression\n4: Polynomial Regression\n: "))
        module_name = modules.get(wh)
        if module_name:
            print("\n Hang Tight. This will take few seconds . . . \n")
            module = importlib.import_module(module_name)
            module.run()
        else:
            print("Invalid choice. Please select a valid option.")

    except ValueError as v:
        print(f'Please enter a valid number.  {v} ')


    c = int(input("conutu ? : "))
    if c == 0 :
        break
