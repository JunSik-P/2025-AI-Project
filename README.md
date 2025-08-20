# 2025-AI-Project
6사단

project/                      # PROJECT_DIR
│
├─ images/                    # 원본 이미지 폴더
│   ├─ bottle/                # 클래스명 폴더 (예: bottle)
│   │    ├─ img001.jpg
│   │    ├─ img002.jpg
│   │    └─ ...
│   ├─ cup/                   # 또 다른 클래스 (예시)
│   │    ├─ img001.jpg
│   │    └─ ...
│   └─ ... (총 10개 클래스 폴더)
│
├─ labels/                    # 라벨 텍스트 저장 폴더
│   ├─ bottle/                # "bottle" 클래스에 대한 샘플 라벨
│   │    ├─ img001.txt        # YOLO 포맷 라벨 파일
│   │    └─ img005.txt
│   ├─ cup/                   # "cup" 클래스에 대한 샘플 라벨
│   │    ├─ img002.txt
│   │    └─ ...
│   └─ ... (총 10개 클래스 폴더)
│
└─ dataset_tmp/               # 자동 생성/임시 저장 (처음엔 비어 있음)
     └─ (코드 실행 시 dataset_tmp_bottle 등 생성됨)
