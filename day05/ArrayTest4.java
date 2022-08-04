package day05;

public class ArrayTest4 {

	public static void main(String[] args) {
		//2차원 배열
		//[1] 선언 [2] 메모리할당 [3] 초기화
		//int [][] a;
		//int a[][];
		int [] a [];//2차원 배열 선언
		
		//[2] 메모리할당 => 3행2열
		a=new int[3][2];
		
		//[3] 초기화
		a[0][0]=10;
		a[0][1]=20;
		
		a[1][0]=30;
		a[1][1]=40;
		
		a[2][0]=50;
		a[2][1]=60;
		
		System.out.println("a[1][1]="+a[1][1]);
		//for루프를 중첩해서 배열 a에 저장된 값을 출력하세요
		//행의 크기 : a.length
		//열의 크기 : a[i].length
		for(int i=0;i<a.length;i++) {
			for(int k=0;k<a[i].length;k++) {
				System.out.print(a[i][k]+" ");
			}
			System.out.println();
		}
		
		System.out.println("a.length: "+a.length);
		System.out.println("a[0].length: "+a[0].length);
		System.out.println("a[1].length: "+a[1].length);
		System.out.println("a[2].length: "+a[2].length);
		
		//배열 a에 저장된 값을 출력하되 확장 for루프문을 이용해 출력해보세요
		for(int aa[] : a)
		{
			for(int aaa : aa)
				System.out.print(aaa+" ");
			System.out.println();
		}
/*	2차원 배열 a가 참조하는 것은 행
 *    a------->행|a[0] | a[1]  | a[2] |
 *    				|    				
 *    			열
 *    			|a[0][0]|a[0][1]|
 *    				
 * 
 * */
		System.out.println("2.-------------");
		//행의 크기만 먼저 잡아놓고, 열의 크기를 나중에 가변적으로 할당할 수 있다.
		char[][]ch=new char[3][];
		ch[0]=new char[2];//2열
		ch[1]=new char[4];//4열
		ch[2]=new char[3];//3열
		
		ch[0][0]='H';
		ch[0][1]='i';
		
		ch[1][0]='J';
		ch[1][1]='a';
		ch[1][2]='v';
		ch[1][3]='a';
		
		ch[2][0]='B';
		ch[2][1]='y';
		ch[2][2]='e';
		
		//for루프 이용해서 ch에 저장된 값을 출력하세요
		for(int i=0;i<ch.length;i++) {
			for(int j=0;j<ch[i].length;j++) {
				System.out.print(ch[i][j]);
			}//for--
			System.out.print(" ");
		}//for----------
		
		for(char[] c:ch) {
			for(char x:c) {
				System.out.print(x);
			}//for---
			System.out.print(" ");
		}//for--------
		System.out.println("\n3.---------------");
		//2차원배열: 선언과 동시에 메모리 할당하고 초기화
		float[]brr[]= {{0.5f,20.1f},{010,011,017},{10,'B'}};
		System.out.println("brr.length: "+brr.length);

		for(float[] f:brr) {
			for(float y:f) {
				System.out.print(y+" ");
			}
			System.out.println();
		}
		
		//double유형의 3차원 배열을 선언과 동시에 메모리할당하고 초기화하세요
		//for루프 돌려서 출력하기
		double []drr[][]= {{{1,2,3},{'a'}},{{10},{20}} };
		for(int i=0;i<drr.length;i++) {
			for(int j=0;j<drr[i].length;j++) {
				for(int k=0;k<drr[i][j].length;k++) {
					System.out.println(drr[i][j][k]);
				}
			}
		}
		
		

	}//main()---------

}//class/////////////////////////////


