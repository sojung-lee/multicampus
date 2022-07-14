package day08;

public class Cafe {

	public static void main(String[] args) {
		CoffeeMachine cm=new CoffeeMachine();
		
		//밀크 커피 만들기
		String s=cm.makeTea(1,2,3);
		System.out.println(s);
		
		System.out.println(cm.makeTea(2, 2, 4));
		
		//설탕 커피
		cm.makeTea(2, 1);
		
		//블랙 커피
		cm.makeTea(1);
		
		//크림 커피
		cm.makeTea(2,(short)3);
		
		//유자차
		Yuja x=new Yuja();
		//x.yuja=3 [x]
		x.setYuja(3);
		x.setSugar(3);
		cm.makeTea(x);
		
	}

}




