try:
    import matplotlib.pyplot as plt
except ImportError:
    import os
    os.system('pip install matplotlib')
    import matplotlib.pyplot as plt
try:
    from prettytable import PrettyTable
except ImportError:
    import os
    os.system('pip install prettytable')
    from prettytable import PrettyTable
from datetime import datetime
import random
import string
now = datetime.now()

# set to the working directory where the files are unloaded.
import os
os.chdir('C:\\Users\\jmori\\Desktop\\DataScience\\Python\\Rental')
# Go back to main menu


def back():
    input('\nPress ENTER to return to main menu.')
    menu()

# Modify vehicle.txt file
def modify_vehicle(tf, mod):
    if tf:
        with open(r'Vehicle.txt', 'w+') as rew:
            for w in mod:
                rew.write(w)

def modify_customer(tf, mod):
    if tf:
        with open(r'Customer.txt', 'w+') as rew:
            for w in mod:
                rew.write(w)

# Display cars available for rent
def display_Vehicle(u):
    table = PrettyTable(['State/Plate', 'Model',
                        'Hourly Rate', 'Weekly Rate', 'Daily Rate', 'Status'])
    with open(r'Vehicle.txt', 'r') as f1:
        cars1 = f1.readlines()
        for i in cars1:
            i = i.split(',')
            i = [x.replace('\n', "") for x in i]
            if u == 'A':
                if i[5] == 'A':
                    table.add_row([i[0], i[1], i[2], i[3], i[4], i[5]])
            elif u == 'R':
                if i[5] == 'R':
                    table.add_row([i[0], i[1], i[2], i[3], i[4], i[5]])
            else:
                print('\nInvalid entry.')
                return
    print(table)

def display_rented_Vehicle():
    table = PrettyTable(['Receipt ID', 'State/Plate', 'Customer ID',
                        'Start DTS', 'End DTS', 'Rental Type', 'Rental Rate'])
    with open(r'rentVehicle.txt', 'r') as f1:
        cars1 = f1.readlines()
        for i in cars1:
            i = i.split(',')
            i = [x.replace('\n', "") for x in i]
            table.add_row([i[0], i[1], i[2], i[3], i[4], i[5], i[6]])
    print(table)

def display_Customers():
    table = PrettyTable(['Customer ID', 'First',
                        'Last', 'Phone', 'Email'])
    with open(r'Customer.txt', 'r') as f1:
        cust1 = f1.readlines()
        for i in cust1:
            i = i.split(',')
            i = [x.replace('\n', "") for x in i]
            table.add_row([i[0], i[1], i[2], i[3], i[4]])
    print(table)

def Add_Customer():
    customer_id    = str(input('Enter customer identifier:'))
    customer_dne = True
    with open(r'Customer.txt', 'r') as cust:
        cus = cust.readlines()
        for ac in cus:
            ac = ac.split(',')
            if customer_id == ac[0]:
                print('\nCustomer exists in database.')
                customer_dne = False
                menu()
                return
    if customer_dne:
        customer_fname = str(input('Enter customer first name:'))
        customer_lname = str(input('Enter customer last name:'))
        customer_phone = str(input('Enter customer phone:'))
        customer_email = str(input('Enter customer email:'))        
        with open(r'Customer.txt', 'a') as fa:
            fa.write('\n'+customer_id + ','+customer_fname+','+customer_lname + ',' +customer_phone+','+customer_email)    
        print("\nCustomer added.")
    while True:
        try:
            con = input('\nDo you want to add another customer? (Y/N): ')
        except Exception:
            print("\nInvalid entry.")
            continue
        else:
            if con == 'Y':
                Add_Customer()
            elif con == 'N':
                menu()
                return
            else:
                print("\nInvalid entry.")
                continue

def Delete_Customer():
    display_Customers()
    customer_id    = str(input('Enter customer identifier to be deleted:'))
    with open(r'Customer.txt', 'r') as crecs:
        ced = crecs.readlines()
        for cbits in ced:
            cbits = cbits.split(',')
            if customer_id == cbits[0]:
                cbits = ','.join(cbits)
                with open(r'Customer.txt', 'a') as custr:
                    custr.write(cbits)
                cpos = ced.index(cbits)
                del ced[cpos]
                modify_customer(True, ced)
                print("\nCustomer deleted.")
                break
        else:
            print("\nCustomer ID not found.")
    while True:
        try:
            con = input('\nDo you want to delete another customer? (Y/N): ')
        except Exception:
            print("\nInvalid entry.")
            continue
        else:
            if con == 'Y' or con == 'y':
                Delete_Customer()
            elif con == 'N' or con == 'n':
                menu()
                return
            else:
                print("\nInvalid entry.")
                continue

# Add or delete vehicles
def Add_Vehicle():
    state_plate = input('\nEnter State/Plate: ')
    vehicle_dne = True
    with open(r'Vehicle.txt', 'r') as fv:
        vec = fv.readlines()
        for vc in vec:
            vc = vc.split(',')
            if state_plate == vc[0]:
                print('\nVehicle exists in database.')
                vehicle_dne = False
    if vehicle_dne:
        model_name = input('Enter Model: ')
        
        while True:
            try:
                hourlyrent = float(input("Enter hourly rent rate: "))
            except Exception:
                print('\nInvalid entry.\n')
                continue
            else:
                hourlyrent = str(hourlyrent)
                break                
        while True:
            try:
                weeklyrent = float(input("Enter weekly rent rate: "))
            except Exception:
                print('\nInvalid entry.\n')
                continue
            else:
                weeklyrent = str(weeklyrent)
                break
        while True:
            try:
                dailyrent = float(input("Enter daily rent rate: "))
            except Exception:
                print('\nInvalid entry.\n')
                continue
            else:
                dailyrent = str(dailyrent)
                break
        stat = input(
            "Enter status (A/R, For R please add rent details to rentVehicle.txt): ")
        with open(r'Vehicle.txt', 'a') as fa:
            fa.write('\n'+state_plate + ','+model_name+','+hourlyrent + ',' +
                     weeklyrent + ','+dailyrent+','+stat)
    while True:
        try:
            con = input('\nDo you want to add another vehicle? (Y/N): ')
        except Exception:
            print("\nInvalid entry.")
            continue
        else:
            if con == 'Y' or con == 'y':
                Add_Vehicle()
            elif con == 'N' or con == 'n':
                menu()
                return
            else:
                print("\nInvalid entry.")
                continue


def Delete_Vehicle():
    state_plate2 = input('\nEnter State/Plate:')
    with open(r'Vehicle.txt', 'r') as fd:
        ved = fd.readlines()
        for vd in ved:
            vd = vd.split(',')
            if state_plate2 == vd[0]:
                vd = ','.join(vd)
                with open(r'deletedVehicles.txt', 'a') as fr:
                    fr.write(vd+'\n')
                pos = ved.index(vd)
                del ved[pos]
                modify_vehicle(True, ved)
                break
        else:
            print("\nVehicle ID not found.")
    while True:
        try:
            #print("\nVehicle deleted.")
            con = input('\nDo you want to delete another vehicle? (Y/N): ')
        except Exception:
            print("\nInvalid entry.")
            continue
        else:
            if con == 'Y' or con == 'y':
                Delete_Vehicle()
            elif con == 'N' or con == 'n':
                menu()
                return
            else:
                print("\nInvalid entry.")
                continue

# Add rent details to files
def rentDetails(path, app):
    plate = app.split(',')
    if path == 1:
        with open(r'rentVehicle.txt', 'a') as q:
            q.write(str(app))
            print('\nCar '+plate[1]+' is rented successfully.')
    elif path == 2:
        with open(r'Transactions.txt', 'a') as p:
            p.write(str(app))
            print('\nCar '+plate[1]+' is returned successfully.')

# Generate Receipt ID and check if it is already taken
def receiptID():
    while True:
        code = ''.join([random.choice(string.ascii_letters + string.digits)
                        for n in range(10)])
        with open('rentVehicle.txt', 'r') as cd:
            cdc = cd.readlines()
            for cdi in cdc:
                cdi = cdi.split(',')
                if cdi[0] == code:
                    continue
            else:
                return code

# Rent a car
def rentVehicle(id1):
    with open(r'Vehicle.txt', 'r') as f2:
        cars2 = f2.readlines()
        cars_temp = cars2.copy()
        for line_no,j in enumerate(cars2):
            j = [i.rstrip() for i in j.split(',')]
            if id1 == j[0]:
                if j[5] == 'A':
                    rentID = input('Enter Customer ID: ')
                    timenow = now.strftime("%m/%d/%Y %H:%M")
                    acc = ' '
                    blank = 'Upon Return'
                    recid = receiptID()
                    print(
                        '\nRental Type Selection:')
                    print('H Hourly')
                    print('D Daily')
                    print('W Weekly')
                    selectrentaltype = input('\nSelect Rental Type? (H,D,W): ')
                    if selectrentaltype == 'H':
                        rental_rate = j[2]
                    elif selectrentaltype == 'D':
                        rental_rate = j[4]
                    elif selectrentaltype == 'W':
                        rental_rate = j[3]
                    #acc = rentaltype()
                    print('\nrentVehicle '+j[0]+' is rented to '+rentID)
                    table1 = PrettyTable(['Receipt ID', recid])
                    table1.add_row(['Customer ID', rentID])
                    table1.add_row(['State/Plate', id1])
                    table1.add_row(['Model', j[1]])
                    table1.add_row(['Status', 'Rented'])
                    table1.add_row(['Rental Type', selectrentaltype])
                    table1.add_row(['Rental Rate', '$'+rental_rate])
                    table1.add_row(['Starting Rental Timestamp',timenow])
                    table1.add_row(['  Ending Rental Timestamp', blank])
                    print(table1)
                    appnew = (recid+','+id1+','+rentID+',' + str(timenow) + ',' + blank + ','+selectrentaltype+',' +rental_rate + '\n')
                    rentDetails(1, appnew)
                    #j[5] = 'A'
                    # problem line
                    #dx = cars2.index(','.join(j))
                    #j[5] = 'R'
                    #j = ','.join(j)
                    #cars2[dx] = j
                    cars_temp[line_no] = j[0] + ',' + j[1] + ',' + j[2] + ',' + j[3] + ',' + j[4] + ',R\n'
                    return True, cars_temp
                elif j[5] == 'R':
                    print('\nVehicle rented.')
                    return False, []
        else:
            print('\nVehicle does not exist.')
            return False, []

# Return, bill, receipt
def generatebill(id2):
    with open(r'rentVehicle.txt', 'r') as f3:
        cars3 = f3.readlines()
        for r in cars3:
            r = r.split(',')
            if id2 == r[0]:
                with open(r'Vehicle.txt', 'r') as f4:
                    rented = f4.readlines()
                    for v in rented:
                        v = v.split(',')
                        if v[0] == r[1]:
                            timenow2 = now.strftime("%m/%d/%Y %H:%M")
                            startdate = r[3]
                            rental_duration = datetime.strptime(timenow2, '%m/%d/%Y %H:%M') - datetime.strptime(startdate, '%m/%d/%Y %H:%M')
                            if r[5] == 'H':
                                temp_rental_hours = rental_duration.total_seconds() // 60 // 60
                                temp_charge = temp_rental_hours * float(r[6])
                                if temp_charge == 0:
                                    charge = float(r[6])
                                    rental_hours = 1
                                else:
                                    rental_hours = temp_rental_hours
                                    charge = temp_charge
                            elif r[5] == 'D':
                                temp_rental_days = rental_duration.total_seconds() // 60 // 60 // 24
                                temp_charge = temp_rental_days * float(r[6])
                                if temp_charge == 0:
                                    charge = float(r[6])
                                    rental_days = 1
                                else:
                                    rental_days = temp_rental_days
                                    charge = temp_charge                                    
                            elif r[5] == 'W':
                                temp_rental_weeks = rental_duration.total_seconds() // 60 // 60 // 24 // 7
                                temp_charge = temp_rental_weeks * float(r[6])  
                                if temp_charge == 0:
                                    charge = float(r[6])  
                                    rental_weeks = 1  
                                else:
                                    rental_weeks = temp_rental_weeks
                                    charge = temp_charge                                      
                            print('\nCar '+r[1]+' is returned from '+r[2])
                            table2 = PrettyTable(['Receipt ID', r[0]])
                            table2.add_row(['Customer ID', r[2]])
                            table2.add_row(['State/Plate', r[1]])
                            table2.add_row(['Description', v[1]])
                            table2.add_row(['Status', 'Returned'])
                            table2.add_row(['Rental Type', ''+r[5].strip('\n')])
                            table2.add_row(['Rental Rate', '$'+r[6].strip('\n')])
                            if r[5] == 'H':
                                table2.add_row(['Duration Hours', str(rental_hours) ])
                            elif r[5] == 'D':
                                table2.add_row(['Duration Days',  str(rental_days) ])
                            elif r[5] == 'W':
                                table2.add_row(['Duration Weeks', str(rental_weeks) ])
                            table2.add_row(['Date/time of rent', r[3]])
                            table2.add_row(['Date/time of return',
                                            now.strftime("%m/%d/%Y %H:%M")])
                            table2.add_row(['Rental charges', '$'+str(charge) ])
                            print(table2)
                            appdone = (id2+','+r[1]+',' + r[2]+',' + r[3] + ','+timenow2+','+str(charge)+'\n')
                            rentDetails(2, appdone)
                            dy = rented.index(','.join(v))
                            v[5] = 'A\n'
                            v = ','.join(v)
                            rented[dy] = v
                            return True, rented
                    else:
                        print(
                            "Receipt ID found but vehicle is not on rent.")
                        return False, []
        else:
            print('\nReceipt ID does not exist.')
            return False, []

# Remove extra lines
def removeblanklines_vehicle():
    with open('Vehicle.txt', 'r') as z:
        content = z.readlines()
        for i in content:
            if i == '\n' or not i:
                content.remove(i)
        modify_vehicle(True, content)

def removeblanklines_customer():
    with open('Customer.txt', 'r') as z:
        content = z.readlines()
        for i in content:
            if i == '\n' or not i:
                content.remove(i)
        modify_customer(True, content)

# Main menu
def menu():
    removeblanklines_vehicle()
    removeblanklines_customer()
    print('\n          Kramerica Kars')
    print('           Vehicle Menu')
    print('Show Vehicle                       1')
    print('Add/Delete Vehicle                 2')
    print('Rent Vehicle                       3')
    print('Generate Billing                   4')
    print('Add/Delete Customer                5')
    print('Show Customer                      6')    
    print('Exit                               7')
    try:
        choice = int(input('\nChoose your options: '))
    except Exception:
        print("\nInvalid entry.")
        back()
    else:
        if choice == 1:
            slt = input(
                '\nA for available, R for rented: ')
            display_Vehicle(slt)
            back()
        elif choice == 2:
            try:
                slt2 = int(
                    input('\nSelect 1 to add 2 to delete vehicles: '))
            except Exception:
                print("\nInvalid entry.")
                back()
            else:
                if slt2 == 1:
                    Add_Vehicle()
                elif slt2 == 2:
                    Delete_Vehicle()
                else:
                    print("\nInvalid entry.")
                    back()
        elif choice == 3:
            slt = 'A'
            display_Vehicle(slt)
            idc2 = input('\nEnter State/Plate: ')
            o2 = rentVehicle(idc2)
            modify_vehicle(o2[0], o2[1])
            back()
        elif choice == 4:
            u = 'R'
            display_rented_Vehicle()
            idc3 = input('\nEnter Receipt ID: ')
            o3 = generatebill(idc3)
            modify_vehicle(o3[0], o3[1])
            back()
        elif choice == 5:
            try:
                slt5 = int(
                    input('\nSelect 1 to add 2 to delete customers: '))
            except Exception:
                print("\nInvalid entry.")
                back()
            else:
                if slt5 == 1:
                    Add_Customer()
                elif slt5 == 2:
                    Delete_Customer()
                else:
                    print("\nInvalid entry.")
                    back()
        elif choice == 6:
            display_Customers()
            back()            
        elif choice == 7:
            print('\nExiting Kramerica Kars.')
            return
        else:
            print('\nInvalid entry.')
            back()

menu()
