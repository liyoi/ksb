import zipfile

# 压缩文件路径
zip_path = 'data.zip'
# 文件存储路径
save_path = '.'


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        print('unzip success')
    else:
        print('This is not zip')


unzip_file(zip_path, save_path)
