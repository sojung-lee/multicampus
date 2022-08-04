package day12;
//매개변수와 리턴값이 없는 람다식
@FunctionalInterface
interface MyFunc{
	void func();
}////////////////////////

//매개변수가 있는 람다식
@FunctionalInterface
interface YourFunc{
	void foo(int x);
}///////////////////////////

//매개변수를 받아 값을 반환하는 람다식
@FunctionalInterface
interface HerFunc{
	String makeStr(String a, String b);
}///////////////////////
public class LambdaTest2 {

	public static void main(String[] args) {
		//[1] 람다식 ###을 출력하는 람다식을 구현한 뒤에 함수를 호출해보세요
		MyFunc mf=()-> System.out.println("###");
		mf.func();
		//[2] 람다식 \\(^^)// 출력하는 람다식 구현하고 함수 호출
		MyFunc mf2=()-> System.out.println("\\\\(^^)//");
		mf2.func();
		//[3] YourFunc  매개변수를 받으면 거듭제곱값을 출력하는 람다식 구현한 뒤 함수 호출하기
		YourFunc yf1=(x) -> System.out.println(x*x);
		yf1.foo(5);
		//[4] YourFunc  매개변수에서 3을 빼주는 값을 출력하는 람다식 구현한 뒤 함수 호출하기
		YourFunc yf2=(int x) -> System.out.println(x-3);
		yf2.foo(10);
		
		//[5] HerFunc  문자열 2개을 받아서 2개의 문자열을 연결해서 반환하는 람다식 구현, 호출하기
		HerFunc hf1=(str1, str2)-> str1+", "+str2;
		System.out.println(hf1.makeStr("hi", "java"));
		
		//[6] HerFunc  문자열 2개을 받아서 2개의 문자열을 연결해서 반환하되 첫글자는 대문자로 나머지는
		//				소문자로 만들어서 반환하는 람다식 구현 호출하기
		// hello,  world ==> Hello World 
		HerFunc hf2=(str1, str2)->{
			char c1=str1.charAt(0);
			char c2=str2.charAt(0);
			String s1=str1.substring(1);
			String s2=str2.substring(1);
			
			String allStr=(c1+"").toUpperCase()+s1+" "+String.valueOf(c2).toUpperCase()+s2;
			return allStr;
		};
		
		System.out.println(hf2.makeStr("hi", "java"));

	}

}
