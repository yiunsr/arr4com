
* cargo build

```
# cargo build for binaray
cargo build --bin arr4combin

# cargo run for binaray
cargo run --bin arr4combin

# cargo build for library
cargo build --lib arr4comlib
```


## cu 파일 ptx 변경
* Visual Studio 의 cl 필요
```
# cmd 실행콘솔에서
cd src\arr4com\cuda\res
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat"  -arch=x64
nvcc cuda_f32.cu -ptx
nvcc cuda_f64.cu -ptx
```

* llvm 으로도 컴파일 가능 한 것 같음 


## unittest 디버깅용 파일생성
```
cargo test
```