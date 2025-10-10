data = {
  'Hugo': 81,
  'Oba': 28,
  'Michael': 90,
  'Cloud': 34,
  'Nica': 59,
  'Jay': 96,
  'Tony': 62,
  'Ruie': 85,
  'Sam': 71,
  'Oscar': 88
}


sum = 0
passed = []
nPassed = []
for single in data:
    sum += data[single]
    if data[single] >= 60:
        passed.append(single)
    else:
        nPassed.append(single)

average = sum / len(data)



print("總平均為:", average)
print(f"及格({len(passed)}人):", ", ".join(passed))
print(f"不及格({len(nPassed)}人):", ", ".join(nPassed))
print(f"及格率:{len(passed)/len(data)*100}%")
if (len(passed) / len(data)*100)>= 50:
    print("通過")
else: print("不通過")