package day02;

public class MyOperator2 {

	public static void main(String[] args) {
		System.out.println("4. 비트연산자 : &, |, ^");
		// 정수에만 사용되는 연산자
		// & (and연산자) : 두 비트가 1일때만 1
		// | (or연산자)  : 두 비트 중 한개라도 1이면 1. 모두 0이면 0
		// ^ (xor연산자) : 두 비트가 서로 다르면 1, 같으면 0
		
		int x=3;  // 0011
		int y=5;  // 0101
				//    
		System.out.println("x&y="+(x&y)); //1
		System.out.println("x|y="+(x|y)); //7
		System.out.println("x^y="+(x^y)); //6
		
		System.out.println("5. 논리 연산자: &, &&, |, ||");
		int a=5;
		int b=6;
		if((a>b) && (a>0)) {
			System.out.println("Hello");
		}else if((b>a)|| (b<0)) {
			System.out.println("Hi");
		}
		/*		
		1) &, &&
		2) |, ||

		&, | 의 경우: &,| 연산자를 사용하면 앞의 연산으로 결과를
					     알 수 있어도 뒤의 문장까지 비교한다.
        &&,|| 의 경우:  앞의 연산으로 결과를 알 수 있을 경우
							뒤의 문장은 수행하지 않는다.
							예를 들어 &&의 경우, 앞의 연산결과가 false이면
							뒤의 연산을 생략.
							||의 경우 앞의 연산결과가 true이면 뒤의 연산을 생략
		*/
		//[문제1]아래 출력문의 결과를 예측하세요
		int i=1; 
		int j=i++;//j=1, i=2
		if((i>++j) && (i++ ==j)) {
			i=i+j;
		}
		System.out.println("i="+i);
		
		//[문제2]
		int k=0;
		int p=1;
		if((k++ ==0)||(p++ ==2)) {
			k=42;
		}
		System.out.println("k="+k+", p="+p);
		//[문제3]
		boolean r=false;
		boolean e=false;
		boolean w=((r=true)||(e=true));
		System.out.println(r+", "+e+", "+w);
		
		
	}

}










