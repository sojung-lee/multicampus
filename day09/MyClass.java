package day09;
//MyInter 를 상속받아서 에러나지 않게 처리해보세요
//인터페이스는 implements로 상속받는다

//YourInter도 상속받아  에러나지 않게 처리해보세요

public class MyClass implements MyInter, YourInter  {
	@Override
	public void demo() {
		System.out.println("MyClass's demo()###");
	}
	public int inter1() {
		
		return 100;
	}
	
	public void inter2(String s) {
		System.out.println(s.toUpperCase());
	}
	
	public static void main(String[] args) {
		//MyInter mi=new MyInter();[x]
		//인터페이는 타입 선언은 할 수 있으나
		//new해서 객체 생성은 할 수 없다.
		MyInter mi = new MyClass();
		YourInter yi = new MyClass();
		
		//demo()
		mi.demo();
		//mi.inter1();[x]
		System.out.println(((MyClass)mi).inter1());
		
		int num=yi.inter1();
		System.out.println("num: "+num);
		yi.inter2("good afternoon");
		//inter1()
		//inter2()
		//PI출력해보기
		
		System.out.println(YourInter.PI);
		System.out.println(MyClass.PI);
		//MyClass.PI=5.12; [x] final변수는 변경 불가능
		System.out.println(yi.PI);
		
		
	}
	

}
