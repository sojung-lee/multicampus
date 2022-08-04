package day12;

class MyObject extends Object{
	@Override
	public String toString() {
		return "I am MyObject";
	}
}///////////////////

public class AnonymousClassTest {

	public static void main(String[] args) {
		Object o1=new Object() ;
		Object o2=new Object() { //이름없는 로컬이너 클래스
			@Override
			public String toString() {
				return "Anonymous Object";
			}
		};
		
		System.out.println(o1);
		System.out.println(o1.toString());
		
		Object o3=new MyObject();
		System.out.println(o3);
		
		System.out.println(o2);

	}

}
