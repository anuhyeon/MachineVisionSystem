present_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
data = []
for i in range(7):
    present_kinematics_pose[i] = i
    data.append(present_kinematics_pose) 
    print(data)

print(data)
print('########')
data2 = []
present_kinematics_pose2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  

for i in range(7):
    present_kinematics_pose2[i] = i 
    data2.append(present_kinematics_pose2[:]) 

print(data2)
