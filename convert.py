# 파일 경로
file_path = '../lane_detection/train/_annotations.txt'

#본인의 데이터셋 경로가 맞는지 검증
new_base_path = '../lane_detection/train/'

# 파일 읽기
with open(file_path, 'r') as file:
    lines = file.readlines()

# 경로 추가
updated_lines = []
for line in lines:
    if line.strip():  
        parts = line.split()  
        image_name = parts[0]  
        new_line = f"{new_base_path}{image_name} {' '.join(parts[1:])}\n"
        updated_lines.append(new_line)
    else:
        updated_lines.append(line)

# 변경된 내용을 파일에 다시 쓰기
with open(file_path, 'w') as file:
    file.writelines(updated_lines)
