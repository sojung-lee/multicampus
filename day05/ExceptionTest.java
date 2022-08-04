package day05;
/* 예외 처리 방법
 * [1] Handle : try~ catch절 사용
 * [2] Declare: throws 절을 사용
 * 
 * 
 * */
public class ExceptionTest {

	public static void main(String[] args) {
		String[] fruits= {"수박","멜론","사과","배"};
		try {
			for(int i=0;i<5;i++) {
				System.out.println(fruits[i]);
			}//for---
		}catch(ArrayIndexOutOfBoundsException e) {
			//예외처리 코드
			System.out.println("배열의 인덱스 범위가 초과되었습니다!!");
		}
			
		System.out.println("###반드시 실행되어야 하는 중요한 코드입니다###");

	}

}
