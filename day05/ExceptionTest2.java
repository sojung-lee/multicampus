package day05;
/*아래 프로그램에서 발생한 오류를 try~catch절로 처리해보세요
 * 단) ExceptionTest2에서는 for루프 바깥쪽에서 try~catch로 잡아보고
 *    ExceptionTest3에서는 for루프문 안쪽에서 try~catch로 잡아보세요
 *   
 * */
public class ExceptionTest2 {

	public static void main(String[] args) {
		try {
			for(int i=1;i<=5;i++) {
				int a=50/(i-3);
			
				System.out.println("a="+a);
			}//for----
		}catch(ArithmeticException e) {
			System.out.println("연산오류!!: 분모가 0이어선 안되어요!!");
		}
		
		System.out.println("##매우 중요한 코드 ####");

	}

}
